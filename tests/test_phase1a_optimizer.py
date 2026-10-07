"""Task 8B Phase 1a F1 fix: tokenizer optimizer + L2-in-Adam regression tests.

The Phase 1a tokenizer collapse was traced to ``torch.optim.Adam(weight_decay=...)``
(L2-in-Adam) shrinking the encoder/latent weight matrices.  These tests lock in the
fix (decoupled AdamW with no-decay groups for LayerNorm/bias/embeddings) and the
mechanism it removes.
"""

import torch

from dreamer import DreamerV4, buildTokenizerOptimizer


def _is_no_decay(name, param):
    return (
        param.ndim <= 1
        or name.endswith(".bias")
        or "norm" in name
        or "embed" in name
        or name in ("mask_token", "patch_pos")
    )


def test_tokenizer_optimizer_is_adamw_with_decoupled_groups(small_dreamer):
    opt = small_dreamer.tokenizerOptimizer
    assert isinstance(opt, torch.optim.AdamW), type(opt)
    wds = sorted({g["weight_decay"] for g in opt.param_groups})
    assert wds == [0.0, 0.01], wds

    wd_by_param = {}
    seen = 0
    for group in opt.param_groups:
        for p in group["params"]:
            wd_by_param[id(p)] = group["weight_decay"]
            seen += 1
    params = list(small_dreamer.tokenizer.named_parameters())
    assert seen == len(params), (seen, len(params))
    for name, p in params:
        if _is_no_decay(name, p):
            assert wd_by_param[id(p)] == 0.0, f"{name} should be no-decay"
        else:
            assert wd_by_param[id(p)] == 0.01, f"{name} should be decayed"


def test_only_tokenizer_optimizer_changed(small_dreamer):
    """Scope decision: the other three optimizers keep Adam+L2 (out of scope)."""
    assert isinstance(small_dreamer.tokenizerOptimizer, torch.optim.AdamW)
    assert type(small_dreamer.dynamicsOptimizer) is torch.optim.Adam
    assert type(small_dreamer.phase2Optimizer) is torch.optim.Adam
    assert type(small_dreamer.phase3Optimizer) is torch.optim.Adam
    assert [g["weight_decay"] for g in small_dreamer.dynamicsOptimizer.param_groups] == [0.01]


def test_l2_in_adam_shrinks_zero_grad_parameter_but_adamw_does_not():
    """Mechanism proof: with a zero data gradient, L2-in-Adam's effective step is
    ``lr * sign(p)`` (shrink ~lr), while decoupled AdamW leaves the parameter
    essentially unchanged.  This is exactly the pathology that killed the encoder."""

    def run(cls):
        p = torch.nn.Parameter(torch.tensor([0.5]))
        opt = cls([p], lr=1e-3, weight_decay=0.01)
        p.grad = torch.zeros_like(p)
        opt.step()
        return float(p.item())

    adam_p = run(torch.optim.Adam)
    adamw_p = run(torch.optim.AdamW)
    assert abs((0.5 - adam_p) - 1e-3) < 1e-4, adam_p          # shrunk by ~lr
    assert (0.5 - adamw_p) < 1e-4, adamw_p                     # decoupled: ~0


def test_buildTokenizerOptimizer_covers_every_parameter(small_dreamer):
    opt = buildTokenizerOptimizer(small_dreamer.tokenizer, lr=1e-4, weight_decay=0.01)
    grouped = [p for g in opt.param_groups for p in g["params"]]
    assert len(grouped) == len(list(small_dreamer.tokenizer.parameters()))
    assert len({id(p) for p in grouped}) == len(grouped)


def test_optimizer_roundtrip_after_fix(small_dreamer, small_batch, tmp_path):
    """Checkpoint round-trip: 5 modules strict-load, AdamW optimizer state complete."""
    small_dreamer.trainPhase1a(small_batch)
    path = tmp_path / "f1_roundtrip"
    small_dreamer.saveCheckpoint(str(path))

    fresh = DreamerV4(
        (3, 64, 64), 3, [-1.0] * 3, [1.0] * 3, torch.device("cpu"),
        small_dreamer.config,
    )
    fresh.loadCheckpoint(str(path))
    assert isinstance(fresh.tokenizerOptimizer, torch.optim.AdamW)
    assert len(fresh.tokenizerOptimizer.state_dict()["state"]) == len(
        small_dreamer.tokenizerOptimizer.state_dict()["state"]
    )
    for name, p in small_dreamer.tokenizer.named_parameters():
        assert torch.equal(p, dict(fresh.tokenizer.named_parameters())[name])
