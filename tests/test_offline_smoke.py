"""Minimal offline runtime smoke test.

Verifies the full three-stage pipeline can run end-to-end without Gymnasium:
construct a tiny DreamerV4, load a temporary .npz dataset, run one step of
Phase 1a/1b/2, finalize Phase 2, run one Phase 3 step, then save and reload a
checkpoint.  All losses must be finite and the expected parameter groups must
change (or stay frozen) across phases.
"""

import numpy as np
import torch
from attridict import AttriDict

from buffer import ReplayBuffer
from dreamer import DreamerV4


def _tiny_config(action_size=2):
    return AttriDict(
        {
            "batchSize": 2,
            "batchLengthShort": 4,
            "batchLengthLong": 6,
            "contextLength": 2,
            "longBatchEvery": 4,
            "imaginationHorizon": 3,
            "actionBins": 5,
            "imageSize": 8,
            "patchSize": 4,
            "patchChannels": 8,
            "tokenizer": AttriDict(
                {
                    "bottleneckTokens": 32,
                    "bottleneckDim": 8,
                    "latentTokens": 16,
                    "latentDim": 16,
                    "modelDim": 32,
                    "layers": 2,
                    "heads": 2,
                    "mlpRatio": 2,
                    "maskRatioMax": 0.9,
                    "lpipsWeight": 0.2,
                    "singleFrameProb": 0.0,
                }
            ),
            "dynamics": AttriDict(
                {
                    "modelDim": 32,
                    "layers": 2,
                    "heads": 2,
                    "kvHeads": 1,
                    "registers": 2,
                    "actionTokens": 1,
                    "agentTokens": 2,
                    "tauBins": 9,
                    "stepBins": [0.25, 0.5, 1.0],
                    "softCap": 50.0,
                    "dropout": 0.0,
                }
            ),
            "sampleSteps": 4,
            "tauCtx": 0.1,
            "mtpLength": 2,
            "discount": 0.997,
            "lambda_": 0.95,
            "pmpoAlpha": 0.5,
            "pmpoBeta": 0.3,
            "twohot": AttriDict({"bins": 32, "low": -5.0, "high": 5.0}),
            "headHidden": 32,
            "lr": 1e-4,
            "phase2Lr": 3e-5,
            "weightDecay": 0.0,
            "gradientClip": 10.0,
            "gradientNormType": 2,
            "precision": "fp32",
            "lossNormDecay": 0.99,
            "buffer": AttriDict({"capacity": 128}),
        }
    )


def _write_npz(path, n=64, action_size=2, obs_shape=(3, 8, 8)):
    observations = np.random.rand(n, *obs_shape).astype(np.float32)
    next_observations = np.random.rand(n, *obs_shape).astype(np.float32)
    actions = np.random.randn(n, action_size).astype(np.float32)
    rewards = np.random.randn(n, 1).astype(np.float32)
    dones = np.zeros((n, 1), dtype=np.float32)
    relevant = (np.arange(n) % 2).astype(np.float32).reshape(-1, 1)
    np.savez(
        path,
        observations=observations,
        nextObservations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        relevant=relevant,
    )
    return path


def _param_snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _changed(before, after):
    for name, p in after.items():
        if not torch.allclose(p, before[name], atol=1e-6):
            return True
    return False


def _assert_state_dicts_equal(first, second, context):
    """Compare full ``state_dict`` (parameters *and* buffers)."""
    first_sd = first.state_dict()
    second_sd = second.state_dict()
    assert set(first_sd) == set(second_sd), (
        f"{context} state_dict key sets differ: "
        f"only-first={set(first_sd) - set(second_sd)}, "
        f"only-second={set(second_sd) - set(first_sd)}"
    )
    for key in first_sd:
        assert torch.allclose(first_sd[key], second_sd[key], atol=1e-6), (
            f"{context} state_dict mismatch at '{key}'"
        )


def test_offline_full_pipeline_smoke(tmp_path):
    config = _tiny_config()
    path = _write_npz(tmp_path / "data.npz")

    dreamer = DreamerV4(
        (3, 8, 8),
        2,
        [-1.0, -1.0],
        [1.0, 1.0],
        torch.device("cpu"),
        config,
    )

    # Load offline data; no Gym environment is ever created here.
    n = dreamer.buffer.loadOffline(str(path))
    assert n == 64
    dreamer.buffer.validateForSampling(config.batchSize, config.batchLengthShort)
    batch = dreamer.buffer.sample(config.batchSize, config.batchLengthShort)
    assert batch.isRelevant.any() and not batch.isRelevant.all()

    tok_before = _param_snapshot(dreamer.tokenizer)
    dyn_before = _param_snapshot(dreamer.dynamics)
    pol_before = _param_snapshot(dreamer.policyHead)
    rwd_before = _param_snapshot(dreamer.rewardHead)
    val_before = _param_snapshot(dreamer.valueHead)

    # Phase 1a: tokenizer only.
    m1a = dreamer.trainPhase1a(batch)
    assert torch.isfinite(torch.tensor(m1a["tokenizerLoss"]))
    assert _changed(tok_before, _param_snapshot(dreamer.tokenizer))

    # Phase 1b: dynamics only.
    m1b = dreamer.trainPhase1b(batch)
    assert torch.isfinite(torch.tensor(m1b["flowLoss"]))
    assert _changed(dyn_before, _param_snapshot(dreamer.dynamics))

    # Phase 2: dynamics + policy + reward.
    dyn_before2 = _param_snapshot(dreamer.dynamics)
    pol_before2 = _param_snapshot(dreamer.policyHead)
    rwd_before2 = _param_snapshot(dreamer.rewardHead)
    val_before2 = _param_snapshot(dreamer.valueHead)
    tok_before2 = _param_snapshot(dreamer.tokenizer)
    m2 = dreamer.trainPhase2(batch)
    for key in ("flowLoss", "bcLoss", "rewardLoss", "phase2Loss"):
        assert torch.isfinite(torch.tensor(m2[key]))
    assert _changed(dyn_before2, _param_snapshot(dreamer.dynamics))
    assert _changed(pol_before2, _param_snapshot(dreamer.policyHead))
    assert _changed(rwd_before2, _param_snapshot(dreamer.rewardHead))
    assert not _changed(val_before2, _param_snapshot(dreamer.valueHead))
    assert not _changed(tok_before2, _param_snapshot(dreamer.tokenizer))

    # Finalize Phase 2 -> frozen prior.
    dreamer.finalizePhase2()
    assert dreamer.frozenPrior is not None

    # Phase 3: policy + value only; dynamics/tokenizer frozen.
    dyn_before3 = _param_snapshot(dreamer.dynamics)
    tok_before3 = _param_snapshot(dreamer.tokenizer)
    pol_before3 = _param_snapshot(dreamer.policyHead)
    val_before3 = _param_snapshot(dreamer.valueHead)
    m3 = dreamer.trainPhase3(batch)
    for key in ("phase3Loss", "pmpoloss", "valueLoss"):
        assert torch.isfinite(torch.tensor(m3[key]))
    assert not _changed(dyn_before3, _param_snapshot(dreamer.dynamics))
    assert not _changed(tok_before3, _param_snapshot(dreamer.tokenizer))
    assert _changed(pol_before3, _param_snapshot(dreamer.policyHead))
    assert _changed(val_before3, _param_snapshot(dreamer.valueHead))

    # Checkpoint save / reload.
    ckpt = tmp_path / "ckpt"
    dreamer.saveCheckpoint(str(ckpt))

    reloaded = DreamerV4(
        (3, 8, 8),
        2,
        [-1.0, -1.0],
        [1.0, 1.0],
        torch.device("cpu"),
        config,
    )
    reloaded.loadCheckpoint(str(ckpt))

    # Full state_dict comparison (parameters *and* buffers), not just
    # named_parameters(), so the RoPE buffers are covered too.
    _assert_state_dicts_equal(dreamer.dynamics, reloaded.dynamics, "dynamics")
    assert reloaded.frozenPrior is not None
    _assert_state_dicts_equal(dreamer.frozenPrior, reloaded.frozenPrior, "frozenPrior")

    # The eight RoPE buffers must be present and on the same device as the
    # reloaded dynamics parameters.
    rope_keys = {
        "backbone.rope_space.cos_t",
        "backbone.rope_space.sin_t",
        "backbone.rope_space.cos_s",
        "backbone.rope_space.sin_s",
        "backbone.rope_time.cos_t",
        "backbone.rope_time.sin_t",
        "backbone.rope_time.cos_s",
        "backbone.rope_time.sin_s",
    }
    reloaded_dynamics_sd = reloaded.dynamics.state_dict()
    assert rope_keys <= set(reloaded_dynamics_sd)
    param_device = next(reloaded.dynamics.parameters()).device
    for key in rope_keys:
        assert reloaded_dynamics_sd[key].device == param_device


def test_phase2_finalize_checkpoint_contains_frozen_prior(tmp_path):
    config = _tiny_config()
    path = _write_npz(tmp_path / "data.npz")
    dreamer = DreamerV4((3, 8, 8), 2, [-1.0, -1.0], [1.0, 1.0], torch.device("cpu"), config)
    dreamer.buffer.loadOffline(str(path))
    batch = dreamer.buffer.sample(config.batchSize, config.batchLengthShort)
    dreamer.trainPhase2(batch)
    dreamer.finalizePhase2()
    assert dreamer.frozenPrior is not None

    ckpt = tmp_path / "final"
    dreamer.saveCheckpoint(str(ckpt))
    saved = torch.load(str(ckpt) + ".pth", map_location="cpu")
    assert saved["frozenPrior"] is not None
    assert len(saved["frozenPrior"]) > 0


def test_phase2_final_checkpoint_phase3_roundtrip(tmp_path):
    config = _tiny_config()
    path = _write_npz(tmp_path / "data.npz")
    source = DreamerV4((3, 8, 8), 2, [-1.0, -1.0], [1.0, 1.0], torch.device("cpu"), config)
    source.buffer.loadOffline(str(path))
    batch = source.buffer.sample(config.batchSize, config.batchLengthShort)
    source.trainPhase2(batch)
    source.finalizePhase2()
    ckpt = tmp_path / "final"
    source.saveCheckpoint(str(ckpt))

    reloaded = DreamerV4((3, 8, 8), 2, [-1.0, -1.0], [1.0, 1.0], torch.device("cpu"), config)
    reloaded.loadCheckpoint(str(ckpt))
    assert reloaded.frozenPrior is not None
    _assert_state_dicts_equal(source.frozenPrior, reloaded.frozenPrior, "frozenPrior")

    metrics = reloaded.trainPhase3(batch)
    for key in ("phase3Loss", "pmpoloss", "valueLoss", "advantages"):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_offline_phase3_requires_prior(tmp_path):
    config = _tiny_config()
    path = _write_npz(tmp_path / "data.npz")
    dreamer = DreamerV4((3, 8, 8), 2, [-1.0, -1.0], [1.0, 1.0], torch.device("cpu"), config)
    dreamer.buffer.loadOffline(str(path))
    batch = dreamer.buffer.sample(config.batchSize, config.batchLengthShort)
    import pytest

    with pytest.raises(RuntimeError):
        dreamer.trainPhase3(batch)
