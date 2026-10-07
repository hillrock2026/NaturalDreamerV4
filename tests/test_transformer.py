import pytest
import torch

from transformer import Attention, RoPE2D, TransformerBackbone, apply_soft_capping


def test_attention_output_shape():
    attn = Attention(dim=64, heads=4, kv_heads=2, soft_cap=50.0)
    x = torch.randn(2, 10, 64)
    y = attn(x)
    assert y.shape == x.shape


def test_gqa_kv_repeat():
    attn = Attention(dim=64, heads=4, kv_heads=2, soft_cap=50.0)
    x = torch.randn(2, 10, 64)
    k = attn.k_proj(x).view(2, 10, attn.kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(2, 10, attn.kv_heads, attn.head_dim).transpose(1, 2)
    rep = attn.heads // attn.kv_heads
    assert k.repeat_interleave(rep, dim=1).shape == (2, attn.heads, 10, attn.head_dim)
    assert v.repeat_interleave(rep, dim=1).shape == (2, attn.heads, 10, attn.head_dim)


def test_soft_capping_bounded():
    logits = torch.tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    capped = apply_soft_capping(logits, cap=50.0)
    assert capped.abs().max().item() <= 50.0


def test_block_space_time_kinds():
    backbone = TransformerBackbone(dim=32, layers=12, heads=2, kv_heads=1, time_every=4)
    kinds = [block.kind for block in backbone.blocks]
    assert len(kinds) == 12
    assert kinds[3] == "time"
    assert kinds[7] == "time"
    assert kinds[11] == "time"
    for i in range(12):
        if i in (3, 7, 11):
            continue
        assert kinds[i] == "space"


def test_backbone_forward_no_error():
    torch.manual_seed(0)
    backbone = TransformerBackbone(dim=64, layers=2, heads=2, kv_heads=1)
    T, S = 2, 4
    x = torch.randn(2, T * S, 64)
    y = backbone(x, T, S)
    assert y.shape == x.shape


def test_backbone_backward():
    torch.manual_seed(0)
    backbone = TransformerBackbone(dim=64, layers=2, heads=2, kv_heads=1)
    T, S = 2, 4
    x = torch.randn(2, T * S, 64, requires_grad=True)
    y = backbone(x, T, S)
    y.sum().backward()
    assert x.grad is not None


def _backbone_forward_backward(backbone, x, T, S, grad_ckpt):
    backbone.gradient_checkpointing = grad_ckpt
    backbone.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    out = backbone(x, T, S)
    out.pow(2).mean().backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in backbone.named_parameters()
        if parameter.grad is not None
    }
    return out.detach().clone(), grads


def test_backbone_gradient_checkpointing_matches_plain_forward_and_grads():
    # Activation checkpointing must not change the forward math (soft capping,
    # masks, GQA, RoPE) or the gradients.  dropout=0 makes the two runs exactly
    # comparable; layers>=4 exercises the time-only blocks too.
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        dim=32, layers=4, heads=2, kv_heads=1, time_every=4, dropout=0.0
    )
    T, S = 3, 4
    x = torch.randn(2, T * S, 32, requires_grad=True)

    with_ckpt_out, with_ckpt_grads = _backbone_forward_backward(backbone, x, T, S, True)
    x_ckpt_grad = x.grad.detach().clone()
    plain_out, plain_grads = _backbone_forward_backward(backbone, x, T, S, False)
    x_plain_grad = x.grad.detach().clone()

    assert with_ckpt_out.shape == plain_out.shape
    assert with_ckpt_out.dtype == plain_out.dtype
    assert torch.allclose(with_ckpt_out, plain_out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(x_ckpt_grad, x_plain_grad, atol=1e-5, rtol=1e-4)
    assert set(with_ckpt_grads) == set(plain_grads)
    for name in with_ckpt_grads:
        assert torch.allclose(with_ckpt_grads[name], plain_grads[name], atol=1e-5, rtol=1e-4), name


def test_backbone_gradient_checkpointing_matches_with_dropout():
    # The formal dynamics config uses dropout=0.1.  Non-reentrant checkpoint
    # preserves the RNG state across recomputation, so with a fixed seed the
    # checkpoint on/off forward and backward must agree.
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        dim=32, layers=4, heads=2, kv_heads=1, time_every=4, dropout=0.1
    )
    backbone.train()
    T, S = 3, 4
    x = torch.randn(2, T * S, 32, requires_grad=True)

    def run(grad_ckpt):
        backbone.gradient_checkpointing = grad_ckpt
        backbone.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        torch.manual_seed(123)
        out = backbone(x, T, S)
        out.pow(2).mean().backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in backbone.named_parameters()
            if parameter.grad is not None
        }
        return out.detach().clone(), x.grad.detach().clone(), grads

    out_on, grad_x_on, grads_on = run(True)
    out_off, grad_x_off, grads_off = run(False)

    assert torch.allclose(out_on, out_off, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_x_on, grad_x_off, atol=1e-5, rtol=1e-4)
    assert set(grads_on) == set(grads_off)
    for name in grads_on:
        assert torch.allclose(grads_on[name], grads_off[name], atol=1e-5, rtol=1e-4), name


def test_backbone_gradient_checkpointing_propagates_to_params_without_input_grad():
    # Phase 2 feeds z1/detached latents and actions that do not require grad;
    # the checkpointed blocks must still produce parameter gradients.
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        dim=32, layers=3, heads=2, kv_heads=1, gradient_checkpointing=True
    )
    backbone.train()
    x = torch.randn(2, 8, 32)  # requires_grad=False
    out = backbone(x, 2, 4)
    assert out.requires_grad, "checkpointed output must require grad via parameters"
    out.pow(2).mean().backward()
    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_backbone_gradient_checkpointing_disabled_still_works():
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        dim=32, layers=2, heads=2, kv_heads=1, gradient_checkpointing=False
    )
    T, S = 2, 4
    x = torch.randn(2, T * S, 32, requires_grad=True)
    out = backbone(x, T, S)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_attention_soft_capping_and_mask_semantics_reference():
    # Documents the exact attention math that checkpointing must preserve:
    # soft-capped logits (before softmax), additive mask, GQA repeat.
    torch.manual_seed(0)
    attn = Attention(dim=16, heads=4, kv_heads=2, soft_cap=1.0)
    x = torch.randn(2, 6, 16)
    B, L, D = x.shape
    out = attn(x)

    q = attn.qknorm(attn.q_proj(x).view(B, L, attn.heads, attn.head_dim).transpose(1, 2))
    k = attn.qknorm(attn.k_proj(x).view(B, L, attn.kv_heads, attn.head_dim).transpose(1, 2))
    v = attn.v_proj(x).view(B, L, attn.kv_heads, attn.head_dim).transpose(1, 2)
    rep = attn.heads // attn.kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scale = 1.0 / (attn.head_dim ** 0.5)
    logits = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    logits = apply_soft_capping(logits, attn.soft_cap)
    weights = torch.softmax(logits, dim=-1)
    reference = torch.einsum("bhij,bhjd->bhid", weights, v).transpose(1, 2).reshape(B, L, D)
    reference = attn.o_proj(reference)

    assert torch.allclose(out, reference, atol=1e-5, rtol=1e-4)


def test_rope2d_is_module_with_cpu_buffers():
    rope = RoPE2D(head_dim=64)
    assert isinstance(rope, torch.nn.Module)
    sd = rope.state_dict()
    for name in ("cos_t", "sin_t", "cos_s", "sin_s"):
        assert name in sd
        assert sd[name].device.type == "cpu"


def test_backbone_to_cpu_moves_rope_buffers():
    backbone = TransformerBackbone(dim=64, layers=2, heads=2, kv_heads=1)
    backbone.to(torch.device("cpu"))
    param_device = next(backbone.parameters()).device
    for rope in (backbone.rope_space, backbone.rope_time):
        for name in ("cos_t", "sin_t", "cos_s", "sin_s"):
            assert getattr(rope, name).device == param_device
    sd = backbone.state_dict()
    assert "rope_space.cos_t" in sd
    assert "rope_time.cos_t" in sd


def test_rope2d_shape_and_finite():
    torch.manual_seed(0)
    B, H, T, S, D = 2, 3, 5, 4, 16
    rope = RoPE2D(head_dim=D)
    q = torch.randn(B, H, T * S, D)
    k = torch.randn(B, H, T * S, D)
    q_out, k_out = rope(q, k, T, S)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()


def test_rope2d_identity_at_position_zero():
    torch.manual_seed(0)
    B, H, T, S, D = 2, 2, 3, 4, 16
    rope = RoPE2D(head_dim=D)
    q = torch.randn(B, H, T * S, D)
    k = torch.randn(B, H, T * S, D)
    q_out, k_out = rope(q, k, T, S)
    assert torch.allclose(q_out[..., 0, :], q[..., 0, :], atol=1e-6)
    assert torch.allclose(k_out[..., 0, :], k[..., 0, :], atol=1e-6)


def test_rope2d_preserves_norm():
    torch.manual_seed(0)
    B, H, T, S, D = 2, 2, 3, 5, 16
    rope = RoPE2D(head_dim=D)
    q = torch.randn(B, H, T * S, D)
    q_out, _ = rope(q, q, T, S)
    assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope2d_distinct_positions_are_rotated():
    torch.manual_seed(0)
    B, H, T, S, D = 1, 1, 3, 2, 16
    rope = RoPE2D(head_dim=D)
    q = torch.randn(B, H, T * S, D)
    q_out, _ = rope(q, q, T, S)
    for idx in range(1, T * S):
        assert not torch.allclose(q_out[..., idx, :], q[..., idx, :], atol=1e-6)


def test_rope2d_q_and_k_share_encoding():
    torch.manual_seed(0)
    B, H, T, S, D = 2, 2, 3, 4, 16
    rope = RoPE2D(head_dim=D)
    q = torch.randn(B, H, T * S, D)
    q_out, k_out = rope(q, q, T, S)
    assert torch.allclose(q_out, k_out, atol=1e-6)


def test_rope2d_time_relative_position():
    torch.manual_seed(0)
    B, H, T, S, D = 1, 1, 4, 1, 16
    rope = RoPE2D(head_dim=D)
    q0 = torch.randn(B, H, 1, D)
    k0 = torch.randn(B, H, 1, D)
    q = q0.expand(B, H, T * S, D)
    k = k0.expand(B, H, T * S, D)
    q_out, k_out = rope(q, k, T, S)
    dots = torch.einsum("bhid,bhjd->bhij", q_out, k_out)
    for delta in range(T * S):
        vals = [dots[0, 0, i, i + delta].item() for i in range(T * S - delta)]
        assert max(vals) - min(vals) < 1e-4


def test_rope2d_rejects_invalid_head_dim():
    for bad in (0, 2, 3, 6, 10):
        with pytest.raises(ValueError):
            RoPE2D(head_dim=bad)


def test_rope2d_accepts_multiple_of_four_head_dim():
    for good in (4, 8, 16, 32, 64):
        rope = RoPE2D(head_dim=good)
        q = torch.randn(1, 1, 3, good)
        q_out, _ = rope(q, q, 3, 1)
        assert q_out.shape == q.shape


def test_rope2d_rejects_length_mismatch():
    # All callers feed RoPE2D through a row-major (B, T*S, D) reshape, so
    # L == T * S is part of the contract; a mismatch has no defined
    # time/space mapping and must be rejected instead of silently accepted.
    rope = RoPE2D(head_dim=16)
    for L, T, S in ((6, 3, 4), (20, 3, 4), (12, 5, 3)):
        q = torch.randn(1, 1, L, 16)
        with pytest.raises(ValueError, match="L == T"):
            rope(q, q, T, S)


def test_rope2d_space_relative_position():
    # T=1 means every token shares the same time position, so only the space
    # rotary should vary; attention logits must depend on the space distance
    # alone.
    torch.manual_seed(0)
    B, H, T, S, D = 1, 1, 1, 5, 16
    rope = RoPE2D(head_dim=D)
    q0 = torch.randn(B, H, 1, D)
    k0 = torch.randn(B, H, 1, D)
    q = q0.expand(B, H, T * S, D)
    k = k0.expand(B, H, T * S, D)
    q_out, k_out = rope(q, k, T, S)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()

    dots = torch.einsum("bhid,bhjd->bhij", q_out, k_out)
    for delta in range(T * S):
        vals = [dots[0, 0, i, i + delta].item() for i in range(T * S - delta)]
        assert max(vals) - min(vals) < 1e-4
    # distinct space positions must not collapse to the same encoding
    assert not torch.allclose(dots[0, 0, 0, 1], dots[0, 0, 0, 0], atol=1e-6)


def test_rope2d_time_and_space_both_vary():
    # A single position change on either axis, and a change on both axes, must
    # each produce distinct rotations (no axis is silently ignored).
    torch.manual_seed(0)
    B, H, T, S, D = 1, 1, 4, 3, 16
    rope = RoPE2D(head_dim=D)
    base = torch.randn(B, H, 1, D).expand(B, H, T * S, D).clone()
    q_out, k_out = rope(base, base, T, S)
    assert q_out.shape == base.shape
    assert torch.isfinite(q_out).all()

    same = q_out[:, :, 0]
    only_space = q_out[:, :, 1]
    only_time = q_out[:, :, S]
    both = q_out[:, :, S + 1]
    assert not torch.allclose(only_space, same, atol=1e-6)
    assert not torch.allclose(only_time, same, atol=1e-6)
    assert not torch.allclose(both, same, atol=1e-6)
    assert not torch.allclose(both, only_space, atol=1e-6)
    assert not torch.allclose(both, only_time, atol=1e-6)
    # identical q and k at identical positions share the same encoding
    assert torch.allclose(q_out, k_out, atol=1e-6)


def test_rope2d_rejects_T_exceeding_max_T():
    rope = RoPE2D(head_dim=16, max_T=4, max_S=8)
    q = torch.randn(1, 1, 5 * 2, 16)
    with pytest.raises(ValueError, match="max_T"):
        rope(q, q, 5, 2)


def test_rope2d_rejects_S_exceeding_max_S():
    rope = RoPE2D(head_dim=16, max_T=8, max_S=4)
    q = torch.randn(1, 1, 2 * 5, 16)
    with pytest.raises(ValueError, match="max_S"):
        rope(q, q, 2, 5)


def test_rope2d_accepts_positions_at_max_bounds():
    rope = RoPE2D(head_dim=16, max_T=5, max_S=4)
    q = torch.randn(1, 1, 5 * 4, 16)
    q_out, k_out = rope(q, q, 5, 4)
    assert q_out.shape == q.shape
    assert torch.isfinite(q_out).all()


def test_attention_rejects_dim_not_divisible_by_heads():
    # dim % heads != 0 would make head_dim = dim // heads silently discard the
    # remainder and shrink the effective model dimension.
    for dim, heads in ((18, 4), (12, 8), (16, 6)):
        with pytest.raises(ValueError, match="divisor"):
            Attention(dim=dim, heads=heads, kv_heads=1)
        with pytest.raises(ValueError, match="divisor"):
            TransformerBackbone(dim=dim, layers=1, heads=heads, kv_heads=1)


def test_backbone_rejects_head_dim_not_multiple_of_four():
    # Distinct from the dim % heads == 0 case: dim is evenly divided, but the
    # resulting head_dim violates RoPE's multiple-of-four requirement.
    with pytest.raises(ValueError, match="multiple of 4"):
        TransformerBackbone(dim=24, layers=1, heads=8, kv_heads=1)


def test_rope2d_q_and_k_independent_inputs_share_encoding():
    torch.manual_seed(0)
    B, H, T, S, D = 2, 3, 4, 3, 16
    rope = RoPE2D(head_dim=D)
    q = torch.randn(B, H, T * S, D)
    k = torch.randn(B, H, T * S, D)
    q_out, k_out = rope(q, k, T, S)
    assert not torch.allclose(q, k)

    idx = torch.arange(T * S)
    steps = idx // S
    pos = idx % S
    cos = torch.cat((rope.cos_t[steps], rope.cos_s[pos]), dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat((rope.sin_t[steps], rope.sin_s[pos]), dim=-1).unsqueeze(0).unsqueeze(0)

    def reference(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    # q and k are independent inputs, so their rotations must differ, yet both
    # are produced from the exact same shared cos/sin table.
    assert not torch.allclose(q_out, k_out)
    assert torch.allclose(q_out, reference(q), atol=1e-6)
    assert torch.allclose(k_out, reference(k), atol=1e-6)


def test_rope2d_bfloat16_input_promotes_to_float32():
    # Buffers are registered in float32 in __init__.  PyTorch type promotion
    # therefore yields a float32 result for a bfloat16 input, matching the
    # elementwise multiply rule (bf16 * fp32 -> fp32).  This documents the
    # observable behaviour; the project does not currently declare a
    # precision contract for RoPE.
    torch.manual_seed(0)
    rope = RoPE2D(head_dim=16)
    assert rope.cos_t.dtype == torch.float32
    q = torch.randn(1, 2, 6, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 6, 16, dtype=torch.bfloat16)
    q_out, k_out = rope(q, k, 2, 3)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == torch.float32
    assert k_out.dtype == torch.float32
    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()


def test_rope2d_bf16_buffers_follow_module_dtype():
    torch.manual_seed(0)
    rope = RoPE2D(head_dim=16).to(torch.bfloat16)
    for name in ("cos_t", "sin_t", "cos_s", "sin_s"):
        assert getattr(rope, name).dtype == torch.bfloat16
    q = torch.randn(1, 2, 6, 16, dtype=torch.bfloat16)
    q_out, k_out = rope(q, q, 2, 3)
    assert q_out.dtype == torch.bfloat16
    assert k_out.dtype == torch.bfloat16
    assert torch.isfinite(q_out).all()


def test_rope2d_autocast_cpu_is_finite_and_matches_fp32():
    torch.manual_seed(0)
    rope = RoPE2D(head_dim=16)
    q = torch.randn(2, 2, 6, 16)
    q_fp32, _ = rope(q, q, 2, 3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        q_amp, k_amp = rope(q, q, 2, 3)
    assert q_amp.shape == q.shape
    assert k_amp.shape == q.shape
    assert torch.isfinite(q_amp).all()
    assert torch.isfinite(k_amp).all()
    assert torch.allclose(q_amp.float(), q_fp32, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rope2d_autocast_cuda_bf16_finite():
    torch.manual_seed(0)
    rope = RoPE2D(head_dim=16).cuda()
    q = torch.randn(2, 2, 6, 16, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        q_out, k_out = rope(q, q, 2, 3)
    assert q_out.shape == q.shape
    assert k_out.shape == q.shape
    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rope2d_cuda_bf16_buffers():
    torch.manual_seed(0)
    rope = RoPE2D(head_dim=16).cuda().to(torch.bfloat16)
    q = torch.randn(2, 2, 6, 16, device="cuda", dtype=torch.bfloat16)
    q_out, k_out = rope(q, q, 2, 3)
    assert q_out.dtype == torch.bfloat16
    assert torch.isfinite(q_out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backbone_cuda_forward():
    backbone = TransformerBackbone(dim=64, layers=2, heads=2, kv_heads=1).cuda()
    T, S = 2, 4
    x = torch.randn(2, T * S, 64, device="cuda")
    y = backbone(x, T, S)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()
    for p in backbone.parameters():
        assert p.device.type == "cuda"
    for rope in (backbone.rope_space, backbone.rope_time):
        for name in ("cos_t", "sin_t", "cos_s", "sin_s"):
            assert getattr(rope, name).device.type == "cuda"
