import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from masks import _to_mask, make_block_causal_mask
from tokenizer import CausalTokenizer


def make_small_tokenizer(small_config):
    return CausalTokenizer(
        small_config.imageSize,
        small_config.patchSize,
        small_config.patchChannels,
        small_config.tokenizer,
    )


def test_encode_shape(small_config):
    tok = make_small_tokenizer(small_config)
    video = torch.randn(2, 4, 3, 64, 64)
    z, mask = tok.encode(video)
    assert z.shape == (2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim)
    assert mask is None  # mask_ratio=0 default


def test_encode_shape_with_mask(small_config):
    tok = make_small_tokenizer(small_config)
    video = torch.randn(2, 4, 3, 64, 64)
    z, mask = tok.encode(video, mask_ratio=0.9)
    assert z.shape == (2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim)
    assert mask.shape == (2, 4, tok.num_patches)
    assert mask.dtype == torch.bool


def test_decode_shape(small_config):
    tok = make_small_tokenizer(small_config)
    z = torch.randn(2, 4, tok.latent_tokens, tok.latent_dim)
    recon = tok.decode(z)
    assert recon.shape == (2, 4, 3, 64, 64)


def test_tokenizer_decode_strict_causal_bitexact(small_config):
    """Perturbing future-frame latents must not change past reconstructions.

    With a strictly block-causal patch self-attention mask the future keys are
    ``-inf`` and contribute exactly zero, so the affected past frames are
    bit-identical (``torch.equal``), not merely close.
    """
    tok = make_small_tokenizer(small_config)
    tok.eval()
    for T in (2, 3, 4):
        torch.manual_seed(1000 + T)
        z = torch.randn(1, T, tok.latent_tokens, tok.latent_dim)
        with torch.no_grad():
            base = tok.decode(z)
        for t in range(1, T):
            z_future = z.clone()
            torch.manual_seed(2000 + 10 * T + t)
            z_future[:, t:] = torch.randn_like(z_future[:, t:])
            with torch.no_grad():
                alt = tok.decode(z_future)
            delta = (alt[:, :t] - base[:, :t]).abs().max().item()
            assert torch.equal(alt[:, :t], base[:, :t]), (
                f"future frame latent leaked into past recon: T={T} t={t} "
                f"max|delta|={delta:.6e}"
            )


def test_tokenizer_decode_past_frames_visible(small_config):
    """Past frames must remain visible: perturbing frame 0 changes frame 0 but
    more importantly the later frames whose causal context includes it."""
    tok = make_small_tokenizer(small_config)
    tok.eval()
    T = 3
    torch.manual_seed(11)
    z = torch.randn(1, T, tok.latent_tokens, tok.latent_dim)
    with torch.no_grad():
        base = tok.decode(z)
    z_past = z.clone()
    torch.manual_seed(12)
    z_past[:, 0] = torch.randn_like(z_past[:, 0])
    with torch.no_grad():
        alt = tok.decode(z_past)
    # The current/last frame may see the perturbed frame 0.
    assert not torch.allclose(alt[:, -1], base[:, -1]), (
        "last frame recon did not change when frame 0 changed: the decoder "
        "cannot see past context (space-only or all -inf mask)"
    )
    # And perturbing a past frame must not alter a future-only sub-block? It
    # legitimately does; the strict direction is future->past (covered above).
    assert not torch.allclose(alt[:, 0], base[:, 0])


def test_tokenizer_decode_tgt_mask_is_block_causal(small_config, monkeypatch):
    tok = make_small_tokenizer(small_config)
    T = 4
    z = torch.randn(1, T, tok.latent_tokens, tok.latent_dim)
    captured = {}

    def fake_decoder(tgt, memory, tgt_mask=None, memory_mask=None, **kwargs):
        captured["tgt_mask"] = tgt_mask.detach().clone()
        return torch.zeros_like(tgt)

    monkeypatch.setattr(tok.decoder, "forward", fake_decoder)
    tok.decode(z)

    expected = make_block_causal_mask(T, tok.num_patches, z.device)
    assert torch.equal(captured["tgt_mask"], expected)
    assert torch.all(torch.diagonal(captured["tgt_mask"]) == 0.0)


def test_phase1a_training_step_finite(small_dreamer, small_batch):
    """Guard: the mask change must keep a Phase 1a step finite."""
    metrics = small_dreamer.trainPhase1a(small_batch)
    for key, value in metrics.items():
        assert math.isfinite(value), f"non-finite metric {key}={value}"


def test_bottleneck_conservation(small_config):
    tok = make_small_tokenizer(small_config)
    assert tok.bottleneck_tokens * tok.bottleneck_dim == tok.latent_tokens * tok.latent_dim
    assert tok.bottleneck_tokens * tok.bottleneck_dim == 2048


def test_mae_per_patch_independent(small_config, monkeypatch):
    tok = make_small_tokenizer(small_config)
    video = torch.rand(1, 1, 3, 64, 64)
    p = torch.full((1, 1, 1, 1), 1.0)
    noise = torch.zeros(1, 1, tok.num_patches, 1)
    half = tok.num_patches // 2
    noise[:, :, :half] = 0.25
    noise[:, :, half:] = 0.75
    rand_calls = iter([p, noise])
    captured_masks = []
    original_where = torch.where

    def fake_rand(*shape, **kwargs):
        return next(rand_calls)

    def fake_where(condition, x, y):
        captured_masks.append(condition.detach().clone())
        return original_where(condition, x, y)

    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(torch, "where", fake_where)
    tok.encode(video, mask_ratio=0.5)

    mask = next(m for m in captured_masks if m.shape == (1, 1, tok.num_patches, 1))
    assert mask.shape == (1, 1, tok.num_patches, 1)
    assert mask.unique().numel() > 1
    assert mask[:, :, :half].all()
    assert not mask[:, :, half:].any()


def test_mae_p_zero_means_no_mask(small_config, monkeypatch):
    tok = make_small_tokenizer(small_config)
    video = torch.rand(1, 1, 3, 64, 64)

    def fail_rand(*shape, **kwargs):
        raise AssertionError("mask random noise should not be sampled when mask_ratio=0")

    monkeypatch.setattr(torch, "rand", fail_rand)
    z, mask = tok.encode(video, mask_ratio=0.0)
    assert mask is None


def test_mae_p_covers_full_range():
    torch.manual_seed(0)
    p = torch.rand(1000, 1, 1, 1) * 0.9
    assert p.min().item() < 0.02
    assert p.max().item() > 0.88


def test_memory_mask_shape_and_causal(small_config, monkeypatch):
    tok = make_small_tokenizer(small_config)
    T = 4
    z = torch.randn(1, T, tok.latent_tokens, tok.latent_dim)
    captured = {}

    def fake_decoder(tgt, memory, tgt_mask=None, memory_mask=None, **kwargs):
        captured["memory_mask"] = memory_mask.detach().clone()
        return torch.zeros_like(tgt)

    monkeypatch.setattr(tok.decoder, "forward", fake_decoder)
    tok.decode(z)

    mask = captured["memory_mask"]
    assert mask.shape == (T * tok.num_patches, T * tok.bottleneck_tokens)
    q_t = torch.arange(T * tok.num_patches) // tok.num_patches
    k_t = torch.arange(T * tok.bottleneck_tokens) // tok.bottleneck_tokens
    expected = _to_mask(k_t[None, :] <= q_t[:, None])
    assert torch.equal(mask, expected)

    first_patch_row = mask[0]
    assert (first_patch_row[: tok.bottleneck_tokens] == 0.0).all()
    assert (first_patch_row[tok.bottleneck_tokens :] == float("-inf")).all()


def test_forward_recon_loss(small_config):
    tok = make_small_tokenizer(small_config)
    video = torch.randn(1, 2, 3, 64, 64)
    recon, z, mask = tok(video, mask_ratio=0.9)
    loss, metrics = tok.loss(video, recon, z, mask, lossnorms=None)
    assert loss.dim() == 0
    loss.backward()
    assert "mse" in metrics
    assert "lpips" in metrics
    assert "tokenizerLoss" in metrics


def test_single_frame_degradation(small_config):
    tok = make_small_tokenizer(small_config)
    video = torch.randn(1, 1, 3, 64, 64)
    recon, z, mask = tok(video)
    assert recon.shape == video.shape
    assert z.shape == (1, 1, tok.latent_tokens, tok.latent_dim)
    assert mask is None


def test_forward_mask_matches_encoder_mask(small_config, monkeypatch):
    tok = make_small_tokenizer(small_config)
    video = torch.rand(2, 3, 3, 64, 64)
    captured = []
    original_where = torch.where

    def fake_where(cond, x, y):
        if cond.shape == (2, 3, tok.num_patches, 1):
            captured.append(cond.detach().clone())
        return original_where(cond, x, y)

    monkeypatch.setattr(torch, "where", fake_where)
    recon, z, mask = tok(video, mask_ratio=0.9)
    assert mask.shape == (2, 3, tok.num_patches)
    assert len(captured) == 1
    assert torch.equal(mask, captured[0][..., 0])


def test_reconstruction_mse_masked_only(small_config):
    tok = make_small_tokenizer(small_config)
    ps = tok.patch_size
    video = torch.rand(1, 1, 3, 64, 64)
    recon = video.clone()
    mask = torch.zeros(1, 1, tok.num_patches, dtype=torch.bool)
    mask[:, :, 0] = True  # mask patch index 0 -> top-left pixel region

    base = tok._reconstruction_mse(recon, video, mask)
    assert base.item() == 0.0  # recon == video

    # Modify an unmasked patch region (index 1 = row 0, col 1) -> loss unchanged.
    recon_unmasked = recon.clone()
    recon_unmasked[:, :, :, 0:ps, ps:2 * ps] += 10.0
    assert torch.allclose(tok._reconstruction_mse(recon_unmasked, video, mask), base)

    # Modify the masked patch region (index 0) -> loss changes.
    recon_masked = recon.clone()
    recon_masked[:, :, :, 0:ps, 0:ps] += 10.0
    assert not torch.allclose(tok._reconstruction_mse(recon_masked, video, mask), base)


def test_reconstruction_mse_full_frame_when_no_mask(small_config):
    tok = make_small_tokenizer(small_config)
    video = torch.rand(1, 2, 3, 64, 64)
    recon = torch.rand(1, 2, 3, 64, 64)
    got = tok._reconstruction_mse(recon, video, None)
    assert torch.allclose(got, torch.nn.functional.mse_loss(recon, video))


def test_forward_loss_finite_with_mask(small_config):
    tok = make_small_tokenizer(small_config)
    video = torch.randn(2, 2, 3, 64, 64)
    recon, z, mask = tok(video, mask_ratio=0.9)
    loss, metrics = tok.loss(video, recon, z, mask, lossnorms=None)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in tok.named_parameters():
        if p.requires_grad:
            assert p.grad is not None


def _install_stub_torchvision(monkeypatch):
    """Install a minimal fake torchvision whose ``vgg16`` returns a real Module.

    This exercises ``CausalTokenizer._load_lpips``' actual
    assignment/registration path (``self._lpips = vgg``) without downloading
    the real VGG weights.  This is a *mechanism* test, not a real-LPIPS test:
    it verifies PyTorch submodule registration and ``nn.Module.to()``
    propagation using a tiny conv stack.  It also records the requested
    ``weights`` argument so tests can assert the fixed pretrained source.
    """

    class _VGG(nn.Module):
        def __init__(self):
            super().__init__()
            # Plain nn.Sequential so the tokenizer's ``vgg[:8]`` slice works.
            self.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1))

    class _Normalize:
        def __init__(self, mean, std):
            self.mean, self.std = mean, std

    class _Weights:
        IMAGENET1K_V1 = "IMAGENET1K_V1"

    calls = {}

    def _vgg16(weights=None):
        calls["weights"] = weights
        return _VGG()

    tv = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    transforms = types.ModuleType("torchvision.transforms")
    models.vgg16 = _vgg16
    models.VGG16_Weights = _Weights
    transforms.Normalize = _Normalize
    tv.models = models
    tv.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    return calls


def test_lpips_is_registered_submodule_with_stub_vgg(small_config, monkeypatch, real_lpips):
    _install_stub_torchvision(monkeypatch)
    tok = make_small_tokenizer(small_config)
    assert tok._lpips is None

    loaded = tok._load_lpips(torch.device("cpu"))
    assert loaded is not None
    assert isinstance(tok._lpips, nn.Module)
    assert "_lpips" in tok._modules
    assert "_lpips" in dict(tok.named_modules())
    assert all(p.device.type == "cpu" for p in tok._lpips.parameters())

    # Real forward + backward through the registered submodule.
    recon = torch.rand(1, 1, 3, 64, 64, requires_grad=True)
    target = torch.rand(1, 1, 3, 64, 64)
    loss = tok.lpips(recon, target)
    assert loss.device.type == "cpu"
    assert torch.isfinite(loss)
    loss.backward()
    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()

    # ``nn.Module.to()`` (the same _apply() path CUDA uses) must reach _lpips.
    tok.to(torch.device("meta"))
    assert next(tok._lpips.parameters()).device.type == "meta"


def test_lpips_real_cpu_forward(small_config, real_lpips):
    pytest.importorskip(
        "torchvision",
        reason="torchvision not available: real VGG LPIPS cannot be executed",
    )
    tok = make_small_tokenizer(small_config)
    loaded = tok._load_lpips(torch.device("cpu"))
    assert loaded is not None, "real VGG LPIPS failed to load on CPU"
    assert isinstance(tok._lpips, nn.Module)
    assert "_lpips" in dict(tok.named_modules())
    assert all(p.device.type == "cpu" for p in tok._lpips.parameters())

    recon = torch.rand(1, 1, 3, 64, 64, requires_grad=True)
    target = torch.rand(1, 1, 3, 64, 64)
    loss = tok.lpips(recon, target)
    assert loss.device.type == "cpu"
    assert torch.isfinite(loss)
    loss.backward()
    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lpips_cuda_migration_and_forward(small_config, real_lpips):
    pytest.importorskip(
        "torchvision",
        reason="torchvision not available: real VGG LPIPS cannot be executed",
    )
    tok = make_small_tokenizer(small_config)
    tok._load_lpips(torch.device("cpu"))
    assert isinstance(tok._lpips, nn.Module)

    tok.cuda()
    assert all(p.device.type == "cuda" for p in tok._lpips.parameters())

    recon = torch.rand(1, 1, 3, 64, 64, device="cuda", requires_grad=True)
    target = torch.rand(1, 1, 3, 64, 64, device="cuda")
    loss = tok.lpips(recon, target)
    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
    loss.backward()
    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()


def test_lpips_requests_fixed_pretrained_weights(small_config, monkeypatch, real_lpips):
    calls = _install_stub_torchvision(monkeypatch)
    tok = make_small_tokenizer(small_config)
    tok._load_lpips(torch.device("cpu"))
    assert calls["weights"] == "IMAGENET1K_V1"


def test_lpips_excluded_from_state_dict(small_config, real_lpips):
    tok = make_small_tokenizer(small_config)
    tok._load_lpips(torch.device("cpu"))
    assert "_lpips" in tok._modules
    state = tok.state_dict()
    assert not any(k == "_lpips" or k.startswith("_lpips.") for k in state)
    # A parent module must not reintroduce the excluded trunk either.
    parent = nn.Sequential(tok)
    parent_state = parent.state_dict()
    assert not any("_lpips" in k for k in parent_state)


def test_load_state_dict_tolerates_lpips_exclusion(small_config, real_lpips):
    source = make_small_tokenizer(small_config)
    source._load_lpips(torch.device("cpu"))
    state = source.state_dict()

    fresh = make_small_tokenizer(small_config)
    fresh.load_state_dict(state, strict=True)
    assert fresh._lpips is None

    # Loading into a tokenizer whose trunk is already loaded must also work,
    # and must keep the loaded trunk rather than dropping it silently.
    source.load_state_dict(state, strict=True)
    assert "_lpips" in source._modules
    assert isinstance(source._lpips, nn.Module)


def test_load_state_dict_ignores_legacy_lpips_keys(small_config, real_lpips):
    tok = make_small_tokenizer(small_config)
    legacy = dict(tok.state_dict())
    legacy["_lpips.0.weight"] = torch.zeros(3, 3, 3, 3)
    legacy["_lpips.0.bias"] = torch.zeros(3)
    tok.load_state_dict(legacy, strict=True)


def test_load_state_dict_strict_rejects_missing_real_key(small_config):
    tok = make_small_tokenizer(small_config)
    state = tok.state_dict()
    missing = next(iter(state))
    del state[missing]
    with pytest.raises(RuntimeError):
        tok.load_state_dict(state, strict=True)


def test_lpips_load_failure_is_explicit(small_config, monkeypatch, real_lpips):
    import torchvision

    def failing_vgg16(*args, **kwargs):
        raise OSError("simulated offline weights cache miss")

    monkeypatch.setattr(torchvision.models, "vgg16", failing_vgg16)
    tok = make_small_tokenizer(small_config)
    with pytest.raises(RuntimeError) as excinfo:
        tok._load_lpips(torch.device("cpu"))
    assert tok._lpips is None
    assert "LPIPS" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)

    # The failure must not be swallowed into a silent zero perceptual loss.
    with pytest.raises(RuntimeError):
        tok.lpips(torch.rand(1, 1, 3, 64, 64), torch.rand(1, 1, 3, 64, 64))


def test_lpips_local_weights_path_is_deterministic(small_config, real_lpips, tmp_path):
    pytest.importorskip("torchvision")
    from torchvision import models

    reference = models.vgg16(weights=None)
    weights_file = tmp_path / "vgg16_local.pth"
    torch.save(reference.state_dict(), weights_file)

    small_config.tokenizer.lpipsWeightsPath = str(weights_file)
    tok = make_small_tokenizer(small_config)
    tok._load_lpips(torch.device("cpu"))
    assert tok._lpips is not None
    assert torch.allclose(tok._lpips[0].weight, reference.features[0].weight)


_LPIPS_PROBE = """
import torch
from attridict import AttriDict
from tokenizer import CausalTokenizer

config = AttriDict({
    "bottleneckTokens": 128,
    "bottleneckDim": 16,
    "latentTokens": 64,
    "latentDim": 32,
    "modelDim": 32,
    "layers": 1,
    "heads": 2,
    "mlpRatio": 2,
    "maskRatioMax": 0.9,
    "lpipsWeight": 0.2,
    "singleFrameProb": 0.0,
})
tok = CausalTokenizer(64, 8, 32, config)
torch.manual_seed(7)
recon = torch.rand(1, 1, 3, 64, 64)
target = torch.rand(1, 1, 3, 64, 64)
print(repr(float(tok.lpips(recon, target))))
"""


def test_lpips_cross_process_consistent(real_lpips):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    values = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", _LPIPS_PROBE],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        values.append(proc.stdout.strip().splitlines()[-1])
    assert values[0] == values[1], f"LPIPS differs across processes: {values}"
