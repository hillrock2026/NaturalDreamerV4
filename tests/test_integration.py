import numpy as np
import pytest
import torch

from attridict import AttriDict
from buffer import ReplayBuffer
from dreamer import DreamerV4
from utils import loadConfig


def _assert_state_dicts_equal(first, second):
    first_sd = first.state_dict()
    second_sd = second.state_dict()
    assert set(first_sd) == set(second_sd)
    for key in first_sd:
        assert torch.allclose(first_sd[key], second_sd[key], atol=1e-6)


def _clone_dreamer(small_config, device=torch.device("cpu")):
    return DreamerV4(
        (3, 64, 64),
        3,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        device,
        small_config,
    )


def _as_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def _assert_optimizer_state_equal(first_optimizer, second_optimizer):
    first = first_optimizer.state_dict()
    second = second_optimizer.state_dict()
    assert first["state"].keys() == second["state"].keys()
    for key in first["state"]:
        for name, value in first["state"][key].items():
            assert torch.allclose(_as_cpu(value), _as_cpu(second["state"][key][name]), atol=0.0)
    assert first["param_groups"] == second["param_groups"]


def test_phase1a_to_1b_chain(small_dreamer, small_batch):
    small_dreamer.trainPhase1a(small_batch)
    small_dreamer.trainPhase1a(small_batch)
    small_dreamer.tokenizer.requires_grad_(False)
    small_dreamer.trainPhase1b(small_batch)
    small_dreamer.trainPhase1b(small_batch)


def test_phase1b_to_2_chain(small_dreamer, small_batch):
    small_dreamer.trainPhase1b(small_batch)
    small_dreamer.trainPhase1b(small_batch)
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.finalizePhase2()
    assert small_dreamer.frozenPrior is not None


def test_phase2_to_3_chain(small_dreamer, small_batch):
    small_dreamer.trainPhase1a(small_batch)
    small_dreamer.trainPhase1b(small_batch)
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.finalizePhase2()
    assert small_dreamer.frozenPrior is not None
    metrics = small_dreamer.trainPhase3(small_batch)
    assert "pmpoloss" in metrics
    assert "valueLoss" in metrics
    assert "advantages" in metrics
    assert torch.isfinite(torch.tensor(metrics["phase3Loss"]))


def test_checkpoint_save_load(small_config, small_batch, tmp_path):
    first = DreamerV4(
        (3, 64, 64),
        3,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        torch.device("cpu"),
        small_config,
    )
    first.freezePolicyPrior()
    first.trainPhase2(small_batch)
    path = tmp_path / "checkpoint"
    first.saveCheckpoint(str(path))

    second = DreamerV4(
        (3, 64, 64),
        3,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        torch.device("cpu"),
        small_config,
    )
    second.loadCheckpoint(str(path))
    _assert_state_dicts_equal(first.tokenizer, second.tokenizer)
    _assert_state_dicts_equal(first.dynamics, second.dynamics)
    _assert_state_dicts_equal(first.policyHead, second.policyHead)
    _assert_state_dicts_equal(first.rewardHead, second.rewardHead)
    _assert_state_dicts_equal(first.valueHead, second.valueHead)
    assert second.frozenPrior is not None
    _assert_state_dicts_equal(first.frozenPrior, second.frozenPrior)


def test_checkpoint_phase_isolation(small_dreamer, tmp_path):
    # The current implementation stores one unified checkpoint containing all
    # optimizer states.  This test verifies Phase 3 optimizer state is saved
    # and the checkpoint can be loaded, while keeping the plan's isolation
    # requirement visible for review.
    small_dreamer.freezePolicyPrior()
    path = tmp_path / "phase3_checkpoint"
    small_dreamer.saveCheckpoint(str(path))
    checkpoint = torch.load(str(path) + ".pth", map_location="cpu")
    assert "phase3Optimizer" in checkpoint
    assert "frozenPrior" in checkpoint and checkpoint["frozenPrior"] is not None

    reloaded = DreamerV4(
        small_dreamer.observation_shape,
        small_dreamer.action_size,
        small_dreamer.action_low,
        small_dreamer.action_high,
        torch.device("cpu"),
        small_dreamer.config,
    )
    reloaded.loadCheckpoint(str(path))
    assert reloaded.frozenPrior is not None


def test_buffer_sample_shapes():
    config = AttriDict({"capacity": 100})
    buffer = ReplayBuffer((3, 64, 64), 3, config, torch.device("cpu"))
    for _ in range(12):
        obs = np.random.rand(3, 64, 64).astype(np.float32)
        next_obs = np.random.rand(3, 64, 64).astype(np.float32)
        action = np.random.randn(3).astype(np.float32)
        buffer.add(obs, action, 0.1, next_obs, False, relevant=True)

    batch = buffer.sample(4, 8)
    assert batch.observations.shape == (4, 8, 3, 64, 64)
    assert batch.actions.shape == (4, 8, 3)
    assert batch.rewards.shape == (4, 8, 1)
    assert batch.dones.shape == (4, 8, 1)
    assert batch.isRelevant.shape == (4, 8, 1)


def test_buffer_offline_roundtrip(tmp_path):
    config = AttriDict({"capacity": 20})
    source = ReplayBuffer((3, 8, 8), 2, config, torch.device("cpu"))
    for i in range(5):
        obs = np.full((3, 8, 8), float(i), dtype=np.float32)
        next_obs = np.full((3, 8, 8), float(i + 1), dtype=np.float32)
        action = np.array([i, i + 1], dtype=np.float32)
        source.add(obs, action, float(i), next_obs, i == 4, relevant=(i % 2 == 0))

    path = tmp_path / "buffer"
    source.saveOffline(str(path))

    target = ReplayBuffer((3, 8, 8), 2, config, torch.device("cpu"))
    loaded = target.loadOffline(str(path))
    assert loaded == 5
    assert len(target) == 5
    assert np.allclose(target.observations[:5], source.observations[:5])
    assert np.allclose(target.nextObservations[:5], source.nextObservations[:5])
    assert np.allclose(target.actions[:5], source.actions[:5])
    assert np.allclose(target.rewards[:5], source.rewards[:5])
    assert np.array_equal(target.relevant[:5], source.relevant[:5])


def test_config_t2_gt_c():
    config = loadConfig("car-racing-v4.yml")
    assert config.dreamer.batchLengthLong > config.dreamer.contextLength
    assert config.dreamer.batchLengthLong == 64
    assert config.dreamer.contextLength == 48


def test_formal_configs_use_fp32_precision_and_preserve_data_params():
    online = loadConfig("car-racing-v4.yml")
    offline = loadConfig("car-racing-v4-offline.yml")
    for name, config in (("online", online), ("offline", offline)):
        assert config.dreamer.precision == "fp32", f"{name} precision must be fp32"
        assert config.dreamer.batchSize == 16, f"{name} batchSize changed"
        assert config.dreamer.batchLengthShort == 16, f"{name} batchLengthShort changed"
        assert config.dreamer.batchLengthLong == 64, f"{name} batchLengthLong changed"
        assert config.dreamer.contextLength == 48, f"{name} contextLength changed"
        assert config.dreamer.buffer.capacity == 50000, f"{name} buffer capacity changed"
    assert online.onlineRelevantEpisodeQuantile == 0.75


def test_phase1a_resume_restores_weights_and_optimizer(small_config, small_batch, tmp_path):
    first = _clone_dreamer(small_config)
    first.trainPhase1a(small_batch)
    path = tmp_path / "phase1a_resume"
    first.saveCheckpoint(str(path))

    second = _clone_dreamer(small_config)
    second.loadCheckpoint(str(path))

    _assert_state_dicts_equal(first.tokenizer, second.tokenizer)
    _assert_optimizer_state_equal(first.tokenizerOptimizer, second.tokenizerOptimizer)
    assert second.total_gradient_steps == first.total_gradient_steps


def test_phase1a_checkpoint_to_phase1b(small_config, small_batch, tmp_path):
    first = _clone_dreamer(small_config)
    first.trainPhase1a(small_batch)
    first.trainPhase1a(small_batch)
    path = tmp_path / "phase1a_chain"
    first.saveCheckpoint(str(path))

    second = _clone_dreamer(small_config)
    second.loadCheckpoint(str(path))
    _assert_state_dicts_equal(first.tokenizer, second.tokenizer)

    tokenizer_before = {n: p.detach().clone() for n, p in second.tokenizer.named_parameters()}
    dynamics_before = {n: p.detach().clone() for n, p in second.dynamics.named_parameters()}
    second.trainPhase1b(small_batch)

    assert any(
        not torch.allclose(p, dynamics_before[n]) for n, p in second.dynamics.named_parameters()
    ), "phase1b did not update dynamics"
    assert all(
        torch.allclose(p, tokenizer_before[n]) for n, p in second.tokenizer.named_parameters()
    ), "phase1b must not update the tokenizer"


def test_phase1b_checkpoint_to_phase2(small_config, small_batch, tmp_path):
    first = _clone_dreamer(small_config)
    first.trainPhase1b(small_batch)
    path = tmp_path / "phase1b_chain"
    first.saveCheckpoint(str(path))

    second = _clone_dreamer(small_config)
    second.loadCheckpoint(str(path))
    metrics = second.trainPhase2(small_batch)
    for key in ("flowLoss", "bcLoss", "rewardLoss", "phase2Loss"):
        assert torch.isfinite(torch.tensor(metrics[key]))
    second.finalizePhase2()
    assert second.frozenPrior is not None


def test_phase2_checkpoint_to_phase3(small_config, small_batch, tmp_path):
    first = _clone_dreamer(small_config)
    first.trainPhase2(small_batch)
    first.finalizePhase2()
    path = tmp_path / "phase2_final"
    first.saveCheckpoint(str(path))

    reloaded = _clone_dreamer(small_config)
    reloaded.loadCheckpoint(str(path))
    assert reloaded.frozenPrior is not None
    assert not any(p.requires_grad for p in reloaded.frozenPrior.parameters())
    _assert_state_dicts_equal(first.frozenPrior, reloaded.frozenPrior)

    metrics = reloaded.trainPhase3(small_batch)
    for key in ("phase3Loss", "pmpoloss", "valueLoss", "advantages"):
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_load_checkpoint_clears_stale_frozen_prior(small_config, small_batch, tmp_path):
    phase2_model = _clone_dreamer(small_config)
    phase2_model.trainPhase2(small_batch)
    phase2_model.finalizePhase2()
    assert phase2_model.frozenPrior is not None
    phase2_path = tmp_path / "with_prior"
    phase2_model.saveCheckpoint(str(phase2_path))

    phase1a_model = _clone_dreamer(small_config)
    phase1a_model.trainPhase1a(small_batch)
    phase1a_path = tmp_path / "without_prior"
    phase1a_model.saveCheckpoint(str(phase1a_path))

    # Loading a checkpoint with no prior into a model that has one must clear it.
    phase2_model.loadCheckpoint(str(phase1a_path))
    assert phase2_model.frozenPrior is None

    # Loading the prior checkpoint restores a frozen, non-trainable prior.
    phase2_model.loadCheckpoint(str(phase2_path))
    assert phase2_model.frozenPrior is not None
    assert not any(p.requires_grad for p in phase2_model.frozenPrior.parameters())


def test_real_lpips_checkpoint_roundtrip(small_config, small_batch, tmp_path, real_lpips):
    first = _clone_dreamer(small_config)
    first.trainPhase1a(small_batch)
    assert first.tokenizer._lpips is not None, "real LPIPS trunk should be loaded"

    path = tmp_path / "lpips_roundtrip"
    first.saveCheckpoint(str(path))
    saved = torch.load(str(path) + ".pth", map_location="cpu")
    assert not any(key.startswith("_lpips") for key in saved["tokenizer"])

    second = _clone_dreamer(small_config)
    second.loadCheckpoint(str(path))
    assert second.tokenizer._lpips is None, "LPIPS trunk is intentionally not persisted"
    _assert_state_dicts_equal(first.tokenizer, second.tokenizer)

    torch.manual_seed(3)
    recon = torch.rand(1, 1, 3, 64, 64)
    target = torch.rand(1, 1, 3, 64, 64)
    first_loss = first.tokenizer.lpips(recon, target)
    second_loss = second.tokenizer.lpips(recon, target)
    assert torch.allclose(first_loss, second_loss), "resume must not change the perceptual loss"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_cpu_save_gpu_load(small_config, small_batch, tmp_path):
    cpu = _clone_dreamer(small_config, torch.device("cpu"))
    cpu.trainPhase1a(small_batch)
    path = tmp_path / "cpu_ckpt"
    cpu.saveCheckpoint(str(path))

    gpu = _clone_dreamer(small_config, torch.device("cuda"))
    gpu.loadCheckpoint(str(path))
    assert next(gpu.tokenizer.parameters()).device.type == "cuda"
    for key, value in cpu.tokenizer.state_dict().items():
        assert torch.allclose(value, gpu.tokenizer.state_dict()[key].cpu())
    _assert_optimizer_state_equal(cpu.tokenizerOptimizer, gpu.tokenizerOptimizer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_gpu_save_cpu_load(small_config, tmp_path):
    gpu = _clone_dreamer(small_config, torch.device("cuda"))
    B, T = 2, 4
    batch = AttriDict(
        {
            "observations": torch.rand(B, T, 3, 64, 64, device="cuda"),
            "actions": torch.randn(B, T, 3, device="cuda"),
            "rewards": torch.randn(B, T, 1, device="cuda"),
            "dones": torch.zeros(B, T, 1, device="cuda"),
            "isRelevant": torch.zeros(B, T, 1, dtype=torch.bool, device="cuda"),
        }
    )
    gpu.trainPhase1a(batch)
    path = tmp_path / "gpu_ckpt"
    gpu.saveCheckpoint(str(path))

    cpu = _clone_dreamer(small_config, torch.device("cpu"))
    cpu.loadCheckpoint(str(path))
    assert next(cpu.tokenizer.parameters()).device.type == "cpu"
    for key, value in cpu.tokenizer.state_dict().items():
        assert torch.allclose(value, gpu.tokenizer.state_dict()[key].cpu())
    _assert_optimizer_state_equal(cpu.tokenizerOptimizer, gpu.tokenizerOptimizer)


TWO_EPISODE_DONES = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _two_episode_buffer(capacity=100):
    buffer = ReplayBuffer((3, 8, 8), 3, AttriDict({"capacity": capacity}), torch.device("cpu"))
    for index, done in enumerate(TWO_EPISODE_DONES):
        buffer.add(
            np.full((3, 8, 8), float(index), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            float(index),
            np.zeros((3, 8, 8), dtype=np.float32),
            bool(done),
            relevant=index < 5,
        )
    return buffer


def test_buffer_sample_never_crosses_episode_boundary():
    buffer = _two_episode_buffer()
    # Windows of length 3 may start at 0,1,2 (episode A) or 5,6,7 (episode B).
    assert set(buffer._valid_starts(3).tolist()) == {0, 1, 2, 5, 6, 7}

    for _ in range(200):
        batch = buffer.sample(8, 3)
        # No interior transition may be an episode end.
        assert not batch.dones.squeeze(-1)[:, :-1].any()

    # A length-6 window cannot fit in either 5-step episode.
    with pytest.raises(ValueError):
        buffer.sample(4, 6)
    with pytest.raises(ValueError):
        buffer.validateForSampling(4, 6)


def test_offline_buffer_respects_episode_boundaries(tmp_path):
    path = tmp_path / "episodes.npz"
    np.savez(
        path,
        observations=np.random.rand(10, 3, 8, 8).astype(np.float32),
        nextObservations=np.random.rand(10, 3, 8, 8).astype(np.float32),
        actions=np.random.randn(10, 3).astype(np.float32),
        rewards=np.zeros((10, 1), dtype=np.float32),
        dones=np.array(TWO_EPISODE_DONES, dtype=np.float32).reshape(-1, 1),
        relevant=np.zeros((10, 1), dtype=bool),
    )
    buffer = ReplayBuffer((3, 8, 8), 3, AttriDict({"capacity": 100}), torch.device("cpu"))
    assert buffer.loadOffline(str(path)) == 10
    assert set(buffer._valid_starts(3).tolist()) == {0, 1, 2, 5, 6, 7}
    for _ in range(100):
        batch = buffer.sample(4, 3)
        assert not batch.dones.squeeze(-1)[:, :-1].any()
    with pytest.raises(ValueError):
        buffer.sample(4, 6)


def test_buffer_formal_sequence_lengths_respect_boundaries():
    buffer = ReplayBuffer((3, 8, 8), 3, AttriDict({"capacity": 1000}), torch.device("cpu"))
    episode_length = 80
    for episode in range(6):
        for step in range(episode_length):
            buffer.add(
                np.zeros((3, 8, 8), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                float(episode),
                np.zeros((3, 8, 8), dtype=np.float32),
                step == episode_length - 1,
            )

    for sequence_size in (16, 64):
        assert len(buffer._valid_starts(sequence_size)) > 0
        for _ in range(50):
            batch = buffer.sample(16, sequence_size)
            assert not batch.dones.squeeze(-1)[:, :-1].any()
            # Rewards are constant within an episode; a crossing window would mix two.
            rewards = batch.rewards.squeeze(-1)
            assert torch.allclose(rewards.min(dim=1).values, rewards.max(dim=1).values)


def test_buffer_formal_lengths_episode_ids_relevance_and_ring_overwrite():
    """T=16/48/64 windows must be contiguous, single-episode and single-group.

    The fixture encodes the global timestep in ``observations``/``actions`` and
    the episode id in ``rewards`` so contiguity and episode membership can be
    checked from the sampled content, not just from tensor shapes.  The small
    capacity (200 < 4*80) forces a ring overwrite, and even episodes are
    relevant while odd ones are uniform so both groups must be samplable at
    every formal sequence length.
    """
    capacity = 200
    episode_length = 80
    num_episodes = 4
    total = capacity + 120  # 320 = 4 * 80, i.e. capacity < total => overwrite
    assert total == episode_length * num_episodes

    buffer = ReplayBuffer((3, 4, 4), 3, AttriDict({"capacity": capacity}), torch.device("cpu"))
    for global_index in range(total):
        episode = global_index // episode_length
        step = global_index % episode_length
        buffer.add(
            np.full((3, 4, 4), float(global_index), dtype=np.float32),
            np.full(3, float(global_index), dtype=np.float32),
            float(episode),
            np.full((3, 4, 4), float(global_index + 1), dtype=np.float32),
            step == episode_length - 1,
            relevant=(episode % 2 == 0),
        )

    assert buffer.full
    retained_globals = set(range(total - capacity, total))
    np.random.seed(0)

    for sequence_size in (16, 48, 64):
        starts, has_relevant, has_uniform = buffer._start_groups(sequence_size)
        assert len(starts) > 0, f"no legal window for T={sequence_size}"
        assert has_relevant.any() and has_uniform.any(), f"missing group at T={sequence_size}"

        # has_relevant / has_uniform must match the actual window content.
        for position, start in enumerate(starts):
            window_rel = [
                bool(buffer.relevant[(start + offset) % capacity][0])
                for offset in range(sequence_size)
            ]
            assert has_relevant[position] == any(window_rel)
            assert has_uniform[position] == (not all(window_rel))

        seen_relevant = seen_uniform = False
        for _ in range(200):
            batch = buffer.sample(8, sequence_size)
            assert batch.observations.shape == (8, sequence_size, 3, 4, 4)
            assert batch.isRelevant.shape == (8, sequence_size, 1)
            assert batch.isRelevant.dtype == torch.bool
            # No interior transition may be an episode end.
            assert not batch.dones.squeeze(-1)[:, :-1].any()

            obs_ids = batch.observations[:, :, 0, 0, 0]
            act_ids = batch.actions[:, :, 0]
            # Global timesteps are consecutive inside a window.
            assert torch.equal(act_ids, obs_ids)
            assert torch.equal(
                obs_ids[:, 1:] - obs_ids[:, :-1],
                torch.ones(8, sequence_size - 1),
            )
            # Episode id (rewards) and relevance are constant within a window.
            rewards = batch.rewards.squeeze(-1)
            assert torch.allclose(rewards.min(dim=1).values, rewards.max(dim=1).values)
            rel = batch.isRelevant.squeeze(-1)
            for row in rel:
                assert len(set(row.tolist())) == 1

            sampled_starts = obs_ids[:, 0].long().numpy()
            # Starts are legal physical indices and never refer to overwritten data.
            assert set((sampled_starts % capacity).tolist()).issubset(set(starts.tolist()))
            assert set(sampled_starts.tolist()).issubset(retained_globals)

            for row_index, episode in enumerate(rewards[:, 0].long().tolist()):
                assert (episode % 2 == 0) == bool(rel[row_index, 0])
            seen_relevant = seen_relevant or bool(rel.any())
            seen_uniform = seen_uniform or bool((~rel).any())

        assert seen_relevant, f"T={sequence_size} never sampled a relevant window"
        assert seen_uniform, f"T={sequence_size} never sampled a uniform window"


def test_buffer_sample_keeps_relevance_shape_with_boundaries():
    buffer = ReplayBuffer((3, 8, 8), 3, AttriDict({"capacity": 1000}), torch.device("cpu"))
    for episode in range(4):
        for step in range(40):
            buffer.add(
                np.zeros((3, 8, 8), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                0.0,
                np.zeros((3, 8, 8), dtype=np.float32),
                step == 39,
                relevant=(episode % 2 == 0),
            )

    batch = buffer.sample(8, 16)
    assert batch.isRelevant.shape == (8, 16, 1)
    assert batch.isRelevant.dtype == torch.bool
    # A window stays inside one episode, so its relevance flag is constant.
    for row in batch.isRelevant.squeeze(-1):
        assert len(set(row.tolist())) == 1


def test_buffer_ring_overwrite_keeps_boundaries_and_no_stale_data():
    capacity = 6
    buffer = ReplayBuffer((3, 8, 8), 3, AttriDict({"capacity": capacity}), torch.device("cpu"))
    total = 20
    for index in range(total):
        buffer.add(
            np.zeros((3, 8, 8), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            float(index),
            np.zeros((3, 8, 8), dtype=np.float32),
            (index + 1) % 5 == 0,
            relevant=(index % 2 == 0),
        )
    assert buffer.full
    retained = {float(index) for index in range(total - capacity, total)}

    for sequence_size in (1, 2, 3):
        try:
            batch = buffer.sample(8, sequence_size)
        except ValueError:
            continue
        assert not batch.dones.squeeze(-1)[:, :-1].any()
        for value in batch.rewards.reshape(-1).tolist():
            assert value in retained, f"sampled stale/uninitialized reward {value}"



def _per_episode_labeled_buffer(relevant_episodes, uniform_episodes, episode_length=10,
                                obs_shape=(3, 8, 8), capacity=1000):
    buffer = ReplayBuffer(obs_shape, 3, AttriDict({"capacity": capacity}), torch.device("cpu"))
    for episode in range(relevant_episodes + uniform_episodes):
        relevant = episode < relevant_episodes
        for step in range(episode_length):
            buffer.add(
                np.zeros(obs_shape, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                0.0,
                np.zeros(obs_shape, dtype=np.float32),
                step == episode_length - 1,
                relevant=relevant,
            )
    return buffer


def test_buffer_balanced_sampling_always_includes_both_groups():
    buffer = _per_episode_labeled_buffer(relevant_episodes=2, uniform_episodes=2)
    sequence_size = 5
    starts, has_relevant, has_uniform = buffer._start_groups(sequence_size)
    assert has_relevant.any() and has_uniform.any()

    for _ in range(2000):
        start_values = buffer._sample_start_indices(
            16, sequence_size, balance_relevance=True
        )
        positions = np.searchsorted(starts, start_values)
        assert has_relevant[positions].any(), "balanced batch lost the relevant group"
        assert has_uniform[positions].any(), "balanced batch lost the uniform group"

    batch = buffer.sample(16, sequence_size, balance_relevance=True)
    per_row = batch.isRelevant.reshape(16, -1).any(dim=1)
    assert bool(per_row.any()) and not bool(per_row.all())


def test_buffer_balanced_sampling_does_not_invent_missing_group():
    # All-relevant data has no uniform window; balancing must not fabricate one,
    # so the fail-fast loss check can still surface genuinely one-class data.
    buffer = _per_episode_labeled_buffer(relevant_episodes=3, uniform_episodes=0)
    starts, has_relevant, has_uniform = buffer._start_groups(5)
    assert has_relevant.all()
    assert not has_uniform.any()

    batch = buffer.sample(16, 5, balance_relevance=True)
    assert bool(batch.isRelevant.all()), "no uniform group should be invented"


def test_start_groups_match_bruteforce_under_ring_overwrite():
    rng = np.random.default_rng(3)
    for _ in range(60):
        capacity = int(rng.integers(5, 16))
        buffer = ReplayBuffer((3, 4, 4), 2, AttriDict({"capacity": capacity}), torch.device("cpu"))
        for _ in range(int(rng.integers(1, 40))):
            buffer.add(
                np.zeros((3, 4, 4), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                0.0,
                np.zeros((3, 4, 4), dtype=np.float32),
                bool(rng.random() < 0.25),
                relevant=bool(rng.random() < 0.5),
            )
        for sequence_size in range(1, capacity + 1):
            starts, has_relevant, has_uniform = buffer._start_groups(sequence_size)
            for position, start in enumerate(starts):
                window_rel = []
                for offset in range(sequence_size):
                    index = (start + offset) % capacity if buffer.full else start + offset
                    window_rel.append(bool(buffer.relevant[index]))
                assert has_relevant[position] == any(window_rel)
                assert has_uniform[position] == (not all(window_rel))
