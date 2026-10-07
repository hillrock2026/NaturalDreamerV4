import os

import numpy as np
import pytest
import torch
from attridict import AttriDict

from buffer import ReplayBuffer
from main import (
    buildArgParser,
    checkpointPath,
    checkpointStepSuffix,
    evaluation_enabled,
    load_offline_dataset,
    resolve_mode,
)


def _write_npz(path, n=20, obs_shape=(3, 8, 8), action_size=2, include_relevant=True):
    observations = np.random.rand(n, *obs_shape).astype(np.float32)
    next_observations = np.random.rand(n, *obs_shape).astype(np.float32)
    actions = np.random.randn(n, action_size).astype(np.float32)
    rewards = np.random.randn(n, 1).astype(np.float32)
    dones = np.zeros((n, 1), dtype=np.float32)
    arrays = {
        "observations": observations,
        "nextObservations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }
    if include_relevant:
        arrays["relevant"] = np.arange(n, dtype=np.float32).reshape(-1, 1) % 2
    np.savez(path, **arrays)
    return path


def _make_buffer(capacity=100, obs_shape=(3, 8, 8), action_size=2):
    config = AttriDict({"capacity": capacity})
    return ReplayBuffer(obs_shape, action_size, config, torch.device("cpu"))


class _FakeDreamer:
    def __init__(self, buffer):
        self.buffer = buffer


def test_resolve_mode_online_default():
    assert resolve_mode(AttriDict({})) == "online"


def test_resolve_mode_offline():
    assert resolve_mode(AttriDict({"mode": "offline"})) == "offline"


def test_resolve_mode_override():
    assert resolve_mode(AttriDict({"mode": "online"}), "offline") == "offline"


def test_resolve_mode_invalid_raises():
    with pytest.raises(ValueError):
        resolve_mode(AttriDict({"mode": "imaginary"}))


def test_evaluation_enabled_online_default():
    assert evaluation_enabled(AttriDict({}), "online") is True


def test_evaluation_enabled_offline_default_false():
    assert evaluation_enabled(AttriDict({}), "offline") is False


def test_evaluation_enabled_offline_explicit():
    assert evaluation_enabled(AttriDict({"evaluateCheckpoints": True}), "offline") is True
    assert evaluation_enabled(AttriDict({"evaluateCheckpoints": False}), "offline") is False


def test_peek_offline_shapes(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=10, obs_shape=(3, 8, 8), action_size=2)
    shape, action_size = ReplayBuffer.peekOffline(str(path))
    assert shape == (3, 8, 8)
    assert action_size == 2


def test_peek_offline_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayBuffer.peekOffline(str(tmp_path / "missing.npz"))


def test_load_offline_missing_file_raises(tmp_path):
    buffer = _make_buffer()
    with pytest.raises(FileNotFoundError):
        buffer.loadOffline(str(tmp_path / "missing.npz"))


def test_load_offline_loads_data(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=10, obs_shape=(3, 8, 8), action_size=2)
    buffer = _make_buffer(capacity=50)
    n = buffer.loadOffline(str(path))
    assert n == 10
    assert len(buffer) == 10
    assert buffer.buffer_index == 10


def test_load_offline_relevant_defaults_false(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=10, include_relevant=False)
    buffer = _make_buffer(capacity=50)
    buffer.loadOffline(str(path))
    assert buffer.missing_relevant is True
    assert not buffer.relevant[:10].any()


def test_load_offline_relevant_preserved(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=10, include_relevant=True)
    buffer = _make_buffer(capacity=50)
    buffer.loadOffline(str(path))
    assert buffer.missing_relevant is False
    expected = np.arange(10, dtype=np.float32).reshape(-1, 1) % 2
    assert np.array_equal(buffer.relevant[:10].reshape(-1), expected.reshape(-1).astype(bool))


def test_load_offline_missing_relevant_error_policy(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=10, include_relevant=False)
    buffer = _make_buffer(capacity=50)
    with pytest.raises(ValueError):
        buffer.loadOffline(str(path), missing_relevant_policy="error")


def test_load_offline_invalid_relevant_shape_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        observations=np.zeros((5, 3, 8, 8), dtype=np.float32),
        actions=np.zeros((5, 2), dtype=np.float32),
        rewards=np.zeros((5, 1), dtype=np.float32),
        relevant=np.zeros((5, 3), dtype=np.float32),  # wrong last dim
    )
    buffer = _make_buffer()
    with pytest.raises(ValueError):
        buffer.loadOffline(str(path))


def test_load_offline_invalid_missing_policy_raises(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=10, include_relevant=False)
    buffer = _make_buffer(capacity=50)
    with pytest.raises(ValueError):
        buffer.loadOffline(str(path), missing_relevant_policy="bogus")


def test_load_offline_missing_required_key_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, observations=np.zeros((5, 3, 8, 8), dtype=np.float32))
    buffer = _make_buffer()
    with pytest.raises(ValueError):
        buffer.loadOffline(str(path))


def test_validate_insufficient_data_raises(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=5)
    buffer = _make_buffer(capacity=50)
    buffer.loadOffline(str(path))
    with pytest.raises(ValueError):
        buffer.validateForSampling(batch_size=4, sequence_size=10)


def test_validate_sequence_too_long_raises(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=20)
    buffer = _make_buffer(capacity=15)
    buffer.loadOffline(str(path))
    with pytest.raises(ValueError):
        buffer.validateForSampling(batch_size=4, sequence_size=30)


def test_sample_empty_buffer_raises():
    buffer = _make_buffer()
    with pytest.raises(ValueError):
        buffer.sample(4, 8)


def test_load_offline_dataset_end_to_end(tmp_path):
    path = _write_npz(tmp_path / "data.npz", n=50, obs_shape=(3, 8, 8), action_size=2)
    dreamer = _FakeDreamer(_make_buffer(capacity=100))
    n = load_offline_dataset(dreamer, str(path), batch_size=4, max_sequence_size=16)
    assert n == 50
    batch = dreamer.buffer.sample(4, 16)
    assert batch.observations.shape == (4, 16, 3, 8, 8)
    assert batch.isRelevant.shape == (4, 16, 1)


def test_load_offline_dataset_empty_path_raises(tmp_path):
    dreamer = _FakeDreamer(_make_buffer())
    with pytest.raises(ValueError):
        load_offline_dataset(dreamer, "", batch_size=4, max_sequence_size=16)


def _mini_dreamer_config(image_size=8, patch_size=4, action_size=2):
    return AttriDict(
        {
            "batchSize": 2,
            "batchLengthShort": 4,
            "batchLengthLong": 6,
            "contextLength": 2,
            "longBatchEvery": 4,
            "imaginationHorizon": 3,
            "actionBins": 5,
            "imageSize": image_size,
            "patchSize": patch_size,
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
            "weightDecay": 0.01,
            "gradientClip": 1.0,
            "gradientNormType": 2,
            "precision": "fp32",
            "lossNormDecay": 0.99,
            "buffer": AttriDict({"capacity": 64}),
        }
    )


def _run_config(tmp_path, n=32, action_size=2, obs_shape=(3, 8, 8)):
    path = _write_npz(tmp_path / "data.npz", n=n, obs_shape=obs_shape, action_size=action_size)
    return AttriDict(
        {
            "environmentName": "FakeEnv-v0",
            "runName": "test",
            "seed": 0,
            "phase": "phase1b",
            "phase1aSteps": 0,
            "phase1bSteps": 2,
            "phase2Steps": 0,
            "phase3Steps": 0,
            "checkpointInterval": 100,
            "saveCheckpoints": False,
            "saveMetrics": False,
            "resume": False,
            "checkpointToLoad": "",
            "mode": "offline",
            "offlineDataPath": str(path),
            "evaluateCheckpoints": False,
            "missingRelevantPolicy": "uniform",
            "actionLow": [-1.0] * action_size,
            "actionHigh": [1.0] * action_size,
            "episodesBeforeStart": 1,
            "numEvaluationEpisodes": 1,
            "folderNames": AttriDict(
                {
                    "metricsFolder": "metrics",
                    "plotsFolder": "plots",
                    "checkpointsFolder": "checkpoints",
                    "videosFolder": "videos",
                }
            ),
            "dreamer": _mini_dreamer_config(action_size=action_size),
        }
    )


def test_offline_run_does_not_build_environment(monkeypatch, tmp_path):
    import main as main_mod
    from dreamer import DreamerV4

    config = _run_config(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("build_pixels_environment must not be called in offline mode")

    monkeypatch.setattr(main_mod, "build_pixels_environment", boom)

    def boom_interact(self, *args, **kwargs):
        raise AssertionError("environmentInteraction must not be called in offline mode")

    monkeypatch.setattr(DreamerV4, "environmentInteraction", boom_interact)

    main_mod.run_training(config, "offline")


def test_offline_run_with_evaluate_does_not_eagerly_build(monkeypatch, tmp_path):
    import main as main_mod
    from dreamer import DreamerV4

    config = _run_config(tmp_path)
    config.evaluateCheckpoints = True  # but no checkpoint is reached in 2 steps

    built = []

    def fake_build(name, render_mode=None):
        built.append((name, render_mode))
        raise AssertionError("eval env must be built lazily only at a checkpoint")

    monkeypatch.setattr(main_mod, "build_pixels_environment", fake_build)

    def boom_interact(self, *args, **kwargs):
        raise AssertionError("environmentInteraction must not be called in offline mode")

    monkeypatch.setattr(DreamerV4, "environmentInteraction", boom_interact)

    main_mod.run_training(config, "offline")
    assert built == []


def test_phase2_saves_distinct_final_checkpoint(tmp_path):
    import main as main_mod

    config = _run_config(tmp_path)
    config.phase = "phase2"
    config.phase2Steps = 2
    config.checkpointInterval = 1
    config.saveCheckpoints = True
    config.saveMetrics = False
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")

    main_mod.run_training(config, "offline")

    run_name = f"{config.environmentName}_{config.runName}"
    ckpts = sorted((tmp_path / "ckpts").glob(f"{run_name}_*.pth"))
    finals = [p for p in ckpts if p.name.endswith("_final.pth")]
    non_finals = [p for p in ckpts if not p.name.endswith("_final.pth")]
    assert len(finals) == 1
    assert non_finals, "a training-time checkpoint should still be produced"
    assert finals[0].name not in {p.name for p in non_finals}

    saved = torch.load(str(finals[0]), map_location="cpu")
    assert saved["frozenPrior"] is not None
    assert len(saved["frozenPrior"]) > 0


def test_phase2_finalize_without_saving_still_builds_prior(monkeypatch, tmp_path):
    import main as main_mod
    from dreamer import DreamerV4

    config = _run_config(tmp_path)
    config.phase = "phase2"
    config.phase2Steps = 2
    config.checkpointInterval = 1
    config.saveCheckpoints = False
    config.saveMetrics = False
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")

    seen = []
    original = DreamerV4.finalizePhase2

    def spy(self):
        original(self)
        seen.append(self.frozenPrior is not None)

    monkeypatch.setattr(DreamerV4, "finalizePhase2", spy)

    main_mod.run_training(config, "offline")

    assert seen == [True]
    assert list((tmp_path / "ckpts").glob("*.pth")) == []


def test_online_run_builds_env_and_warms_up(monkeypatch, tmp_path):
    import main as main_mod
    from dreamer import DreamerV4

    config = _run_config(tmp_path)
    config.mode = "online"
    config.phase1bSteps = 0  # stop after setup/warmup

    built = []

    def fake_build(name, render_mode=None):
        built.append((name, render_mode))
        return object()

    monkeypatch.setattr(main_mod, "build_pixels_environment", fake_build)
    monkeypatch.setattr(
        main_mod, "get_env_properties", lambda env: ((3, 8, 8), 2, [-1.0, -1.0], [1.0, 1.0])
    )

    interacted = []

    def fake_interact(self, env, num, seed=None, **kwargs):
        interacted.append(num)
        return 0.0

    monkeypatch.setattr(DreamerV4, "environmentInteraction", fake_interact)

    main_mod.run_training(config, "online")
    assert len(built) == 2  # training env + eager evaluation env
    assert interacted == [1]  # warmup happened


def test_checkpoint_step_suffix_small_and_large():
    assert checkpointStepSuffix(0) == "0steps"
    assert checkpointStepSuffix(1) == "1steps"
    assert checkpointStepSuffix(2) == "2steps"
    assert checkpointStepSuffix(999) == "999steps"
    assert checkpointStepSuffix(1000) == "1k"
    assert checkpointStepSuffix(2000) == "2k"
    assert checkpointStepSuffix(60000) == "60k"


def test_checkpoint_step_suffix_small_steps_are_unique():
    small = [checkpointStepSuffix(n) for n in range(1, 12)]
    assert len(set(small)) == len(small)


def test_small_step_checkpoints_do_not_collide(tmp_path):
    import main as main_mod

    config = _run_config(tmp_path)
    config.phase = "phase1a"
    config.phase1aSteps = 3
    config.checkpointInterval = 1
    config.saveCheckpoints = True
    config.saveMetrics = False
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")

    main_mod.run_training(config, "offline")

    run_name = f"{config.environmentName}_{config.runName}"
    names = sorted(p.name for p in (tmp_path / "ckpts").glob(f"{run_name}_*.pth"))
    assert names == [
        f"{run_name}_1steps.pth",
        f"{run_name}_2steps.pth",
        f"{run_name}_3steps.pth",
    ]


def test_checkpoint_path_uses_run_naming(tmp_path):
    config = _run_config(tmp_path)
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")
    config.checkpointToLoad = "5steps"
    assert checkpointPath(config) == os.path.join(
        str(tmp_path / "ckpts"), f"{config.environmentName}_{config.runName}_5steps"
    )


def test_cli_parser_defaults():
    args = buildArgParser().parse_args([])
    assert args.config == "car-racing-v4.yml"
    assert args.phase is None
    assert args.mode is None
    assert args.resume is None
    assert args.checkpoint_to_load is None


def test_cli_parser_overrides():
    args = buildArgParser().parse_args(
        [
            "--config",
            "car-racing-v4.yml",
            "--phase",
            "phase1b",
            "--mode",
            "offline",
            "--resume",
            "--checkpoint-to-load",
            "5steps",
        ]
    )
    assert args.phase == "phase1b"
    assert args.mode == "offline"
    assert args.resume is True
    assert args.checkpoint_to_load == "5steps"


def test_cli_parser_no_resume():
    args = buildArgParser().parse_args(["--no-resume"])
    assert args.resume is False


def test_main_applies_cli_overrides(monkeypatch):
    import main as main_mod

    captured = {}
    monkeypatch.setattr(
        main_mod, "run_training", lambda config, mode: captured.update(config=config, mode=mode)
    )
    main_mod.main(
        "car-racing-v4.yml",
        "phase1b",
        "offline",
        resume_override=True,
        checkpoint_override="5steps",
    )
    assert captured["config"].phase == "phase1b"
    assert captured["config"].resume is True
    assert captured["config"].checkpointToLoad == "5steps"
    assert captured["mode"] == "offline"


def test_main_checkpoint_override_implies_resume(monkeypatch):
    import main as main_mod

    captured = {}
    monkeypatch.setattr(
        main_mod, "run_training", lambda config, mode: captured.update(config=config)
    )
    main_mod.main("car-racing-v4.yml", "phase1b", "offline", checkpoint_override="5steps")
    assert captured["config"].resume is True
    assert captured["config"].checkpointToLoad == "5steps"


def test_main_no_resume_overrides_yaml_and_checkpoint(monkeypatch):
    import main as main_mod

    captured = {}
    monkeypatch.setattr(
        main_mod, "run_training", lambda config, mode: captured.update(config=config)
    )
    main_mod.main(
        "car-racing-v4.yml",
        "phase1b",
        "offline",
        resume_override=False,
        checkpoint_override="5steps",
    )
    assert captured["config"].resume is False
    assert captured["config"].checkpointToLoad == "5steps"


def test_resume_without_checkpoint_raises(tmp_path):
    import main as main_mod

    config = _run_config(tmp_path)
    config.phase = "phase1b"
    config.phase1bSteps = 1
    config.resume = True
    config.checkpointToLoad = ""
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")

    with pytest.raises(ValueError):
        main_mod.run_training(config, "offline")


def test_resume_missing_checkpoint_file_raises(tmp_path):
    import main as main_mod

    config = _run_config(tmp_path)
    config.phase = "phase1b"
    config.phase1bSteps = 1
    config.resume = True
    config.checkpointToLoad = "99steps"
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")

    with pytest.raises(FileNotFoundError):
        main_mod.run_training(config, "offline")


def test_offline_training_logs_precision_contract(tmp_path, capsys):
    import main as main_mod

    config = _run_config(tmp_path)
    config.phase = "phase1a"
    config.phase1aSteps = 1
    config.saveCheckpoints = False
    config.saveMetrics = False

    main_mod.run_training(config, "offline")

    out = capsys.readouterr().out
    assert "precision: fp32" in out
    assert "model_dtype: torch.float32" in out
    assert "buffer_dtype:" in out
    assert "autocast_enabled: false" in out
    assert "device:" in out


def test_offline_training_rejects_bf16_config(tmp_path):
    import main as main_mod

    config = _run_config(tmp_path)
    config.phase = "phase1a"
    config.phase1aSteps = 1
    config.saveCheckpoints = False
    config.saveMetrics = False
    config.dreamer.precision = "bf16"

    with pytest.raises(ValueError) as excinfo:
        main_mod.run_training(config, "offline")
    assert "bf16" in str(excinfo.value)


def test_training_uses_balanced_sampling_only_for_relevance_phases(monkeypatch, tmp_path):
    import main as main_mod
    from buffer import ReplayBuffer

    recorded = []
    original = ReplayBuffer.sample

    def spy(self, batch_size, sequence_size, balance_relevance=False):
        recorded.append(balance_relevance)
        return original(self, batch_size, sequence_size, balance_relevance=balance_relevance)

    monkeypatch.setattr(ReplayBuffer, "sample", spy)

    config = _run_config(tmp_path)
    config.phase = "phase1a"
    config.phase1aSteps = 1
    main_mod.run_training(config, "offline")
    assert recorded and all(value is False for value in recorded)

    recorded.clear()
    config.phase = "phase2"
    config.phase2Steps = 1
    config.folderNames.checkpointsFolder = str(tmp_path / "ckpts")
    main_mod.run_training(config, "offline")
    assert recorded and all(value is True for value in recorded)
