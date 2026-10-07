import copy

import numpy as np
import pytest
import torch
from attridict import AttriDict
from torch.distributions import Independent

import dreamer as dreamer_mod
from heads import discretize_actions, pmpo_loss as original_pmpo_loss


def _freeze_prior(dreamer):
    dreamer.freezePolicyPrior()


class _FakeActionSpace:
    def sample(self):
        return np.zeros(3, dtype=np.float32)


class _FakeRewardEnv:
    """Fake env with a fixed reward and step count per episode.

    Episode lengths are kept below the fixture ``contextLength`` so the
    random-action warmup branch is used and no model forward is needed.
    """

    def __init__(self, rewards, steps):
        assert len(rewards) == len(steps)
        self._rewards = list(rewards)
        self._steps = list(steps)
        self._episode = 0
        self.action_space = _FakeActionSpace()

    def reset(self, seed=None):
        self._step = 0
        self._reward = self._rewards[self._episode]
        self._length = self._steps[self._episode]
        self._episode += 1
        return np.zeros((3, 64, 64), dtype=np.float32)

    def step(self, action):
        self._step += 1
        done = self._step >= self._length
        observation = np.full((3, 64, 64), self._step, dtype=np.float32)
        return observation, self._reward, done


class _FakeEnv:
    """Single-step-per-episode fake env whose reward equals the reset seed."""

    action_space = _FakeActionSpace()

    def reset(self, seed=None):
        self.reward = float(seed)
        return np.zeros((3, 64, 64), dtype=np.float32)

    def step(self, action):
        return np.full((3, 64, 64), 1.0, dtype=np.float32), self.reward, True


def test_resolve_precision_accepts_fp32():
    assert dreamer_mod.resolvePrecision("fp32") == "fp32"
    assert dreamer_mod.resolvePrecision("FP32") == "fp32"
    assert dreamer_mod.resolvePrecision(None) == "fp32"


@pytest.mark.parametrize(
    "value", ["bf16", "bfloat16", "fp16", "float", "32", "default", ""]
)
def test_resolve_precision_rejects_unsupported_values(value):
    with pytest.raises(ValueError):
        dreamer_mod.resolvePrecision(value)


def test_bf16_precision_config_fails_fast(small_config):
    config = copy.deepcopy(small_config)
    config.precision = "bf16"
    with pytest.raises(ValueError) as excinfo:
        dreamer_mod.DreamerV4(
            (3, 64, 64), 3, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], torch.device("cpu"), config
        )
    assert "bf16" in str(excinfo.value)


def _assert_all_params_fp32(model):
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, f"unexpected parameter dtypes: {dtypes}"
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all(), "parameter became NaN/Inf"


def test_fp32_train_phases_stay_float32_without_autocast(small_dreamer, small_batch):
    assert torch.is_autocast_enabled() is False
    assert small_dreamer.precision == "fp32"
    _assert_all_params_fp32(small_dreamer)

    metrics_1a = small_dreamer.trainPhase1a(small_batch)
    assert torch.isfinite(torch.tensor(metrics_1a["mse"]))
    _assert_all_params_fp32(small_dreamer)

    metrics_1b = small_dreamer.trainPhase1b(small_batch)
    assert torch.isfinite(torch.tensor(metrics_1b["flowLoss"]))
    _assert_all_params_fp32(small_dreamer)

    metrics_2 = small_dreamer.trainPhase2(small_batch)
    for key in ("flowLoss", "bcLoss", "rewardLoss", "phase2Loss"):
        assert torch.isfinite(torch.tensor(metrics_2[key]))
    _assert_all_params_fp32(small_dreamer)

    small_dreamer.finalizePhase2()
    metrics_3 = small_dreamer.trainPhase3(small_batch)
    for key in ("phase3Loss", "pmpoloss", "valueLoss"):
        assert torch.isfinite(torch.tensor(metrics_3[key]))
    _assert_all_params_fp32(small_dreamer)
    assert not torch.is_autocast_enabled()

    # Optimizer state must stay float32 under the fp32 contract.
    for group in small_dreamer.tokenizerOptimizer.state.values():
        for value in group.values():
            if torch.is_tensor(value):
                assert value.dtype == torch.float32


def test_precision_report_reflects_actual_dtypes(small_dreamer):
    report = small_dreamer.precisionReport()
    assert report["precision"] == "fp32"
    assert report["model_dtypes"] == ["torch.float32"]
    assert report["autocast_enabled"] is False
    assert set(report["buffer_dtypes"]) == {"numpy.float32", "numpy.bool"}
    assert report["device"] == "cpu"


def test_phase1a_step(small_dreamer, small_batch):
    metrics = small_dreamer.trainPhase1a(small_batch)
    assert "mse" in metrics
    assert "lpips" in metrics
    assert "psnr" in metrics


def test_phase1b_step(small_dreamer, small_batch):
    metrics = small_dreamer.trainPhase1b(small_batch)
    assert "flowLoss" in metrics


def test_online_interaction_marks_top_reward_episodes_relevant(small_dreamer):
    small_dreamer.environmentInteraction(
        _FakeEnv(),
        num_episodes=4,
        seed=1,
        relevant_episode_quantile=0.5,
    )

    assert small_dreamer.buffer.relevant[:2].sum() == 0
    assert small_dreamer.buffer.relevant[2:4].all()


def test_online_relevance_covers_whole_episodes_and_ranks_returns(small_dreamer):
    rewards = [1.0, 2.0, 3.0, 4.0]
    steps = [3, 3, 3, 3]
    small_dreamer.environmentInteraction(
        _FakeRewardEnv(rewards, steps),
        num_episodes=4,
        seed=1,
        relevant_episode_quantile=0.5,
    )

    relevant = small_dreamer.buffer.relevant[:12].reshape(-1)
    episode_labels = [bool(relevant[i * 3]) for i in range(4)]
    # Every transition of an episode shares the episode's label (full coverage).
    for episode in range(4):
        assert relevant[episode * 3:(episode + 1) * 3].tolist() == [episode_labels[episode]] * 3
    # Top half by return is relevant.
    assert episode_labels == [False, False, True, True]
    assert relevant[9:12].all(), "highest-return episode must be fully relevant"
    assert not relevant[0:3].any(), "lowest-return episode must be fully uniform"
    assert small_dreamer.buffer.relevant[:12].sum() > 0
    assert (~small_dreamer.buffer.relevant[:12]).sum() > 0


def test_online_relevance_quantile_075_marks_top_quarter(small_dreamer):
    # 4 episodes with clearly separated returns; quantile=0.75 means the top
    # ceil(0.25 * 4) = 1 episode (the highest return) is relevant.
    small_dreamer.environmentInteraction(
        _FakeRewardEnv([1.0, 2.0, 3.0, 4.0], [4, 4, 4, 4]),
        num_episodes=4,
        seed=1,
        relevant_episode_quantile=0.75,
    )
    relevant = small_dreamer.buffer.relevant[:16].reshape(-1)
    labels = [bool(relevant[episode * 4]) for episode in range(4)]
    for episode in range(4):
        assert relevant[episode * 4:(episode + 1) * 4].tolist() == [labels[episode]] * 4
    assert labels == [False, False, False, True]
    assert sum(labels) == 1
    assert relevant.any() and (~relevant).any()


def test_online_relevance_quantile_075_marks_two_of_eight(small_config):
    from dreamer import DreamerV4

    dreamer = DreamerV4(
        (3, 64, 64), 3, [-1, -1, -1], [1, 1, 1], torch.device("cpu"), small_config
    )
    dreamer.environmentInteraction(
        _FakeRewardEnv([float(value) for value in range(1, 9)], [2] * 8),
        num_episodes=8,
        seed=1,
        relevant_episode_quantile=0.75,
    )
    relevant = dreamer.buffer.relevant[:16].reshape(-1)
    labels = [bool(relevant[episode * 2]) for episode in range(8)]
    for episode in range(8):
        assert relevant[episode * 2:(episode + 1) * 2].tolist() == [labels[episode]] * 2
    # ceil(0.25 * 8) = 2 highest-return episodes.
    assert labels == [False] * 6 + [True, True]


def test_online_relevance_equal_returns_is_mixed_and_reproducible(small_config):
    from dreamer import DreamerV4

    def run_once():
        dreamer = DreamerV4(
            (3, 64, 64), 3, [-1, -1, -1], [1, 1, 1], torch.device("cpu"), small_config
        )
        dreamer.environmentInteraction(
            _FakeRewardEnv([5.0, 5.0, 5.0, 5.0], [2, 2, 2, 2]),
            num_episodes=4,
            seed=7,
            relevant_episode_quantile=0.5,
        )
        return dreamer.buffer.relevant[:8].reshape(-1).tolist()

    first = run_once()
    second = run_once()
    assert first == second, "tie-breaking must be deterministic"
    assert any(first) and not all(first), "equal returns must not collapse to one class"


@pytest.mark.parametrize("quantile", [-0.1, 1.0, 1.5])
def test_online_relevance_invalid_quantile_raises(small_dreamer, quantile):
    with pytest.raises(ValueError):
        small_dreamer.environmentInteraction(
            _FakeRewardEnv([1.0, 2.0, 3.0, 4.0], [2, 2, 2, 2]),
            num_episodes=4,
            seed=1,
            relevant_episode_quantile=quantile,
        )


def test_online_relevance_requires_at_least_two_episodes(small_dreamer):
    with pytest.raises(ValueError):
        small_dreamer.environmentInteraction(
            _FakeRewardEnv([1.0], [2]),
            num_episodes=1,
            seed=1,
            relevant_episode_quantile=0.5,
        )


def test_online_evaluation_does_not_write_to_buffer(small_dreamer):
    small_dreamer.environmentInteraction(
        _FakeRewardEnv([1.0, 2.0], [2, 2]),
        num_episodes=2,
        seed=1,
        relevant_episode_quantile=0.5,
    )
    before = len(small_dreamer.buffer)
    episodes_before = small_dreamer.total_episodes
    steps_before = small_dreamer.total_env_steps
    relevant_snapshot = small_dreamer.buffer.relevant[:before].copy()

    small_dreamer.environmentInteraction(
        _FakeRewardEnv([9.0], [2]),
        num_episodes=1,
        seed=1,
        evaluation=True,
    )

    assert len(small_dreamer.buffer) == before
    assert small_dreamer.total_episodes == episodes_before
    assert small_dreamer.total_env_steps == steps_before
    assert np.array_equal(small_dreamer.buffer.relevant[:before], relevant_snapshot)


def test_online_total_env_steps_counts_all_steps(small_dreamer):
    small_dreamer.environmentInteraction(
        _FakeRewardEnv([1.0, 2.0, 3.0], [2, 3, 2]),
        num_episodes=3,
        seed=1,
        relevant_episode_quantile=0.5,
    )
    # 2 + 3 + 2 real env steps; the old ``len(history)`` accounting would give 10.
    assert small_dreamer.total_env_steps == 7
    assert small_dreamer.total_episodes == 3


def test_real_carracing_online_warmup_marks_relevant(small_config):
    from dreamer import DreamerV4
    from main import build_pixels_environment

    try:
        env = build_pixels_environment("CarRacing-v3")
    except RuntimeError as exc:  # pragma: no cover - only when gym is unavailable
        pytest.skip(f"CarRacing-v3 unavailable: {exc}")

    config = copy.deepcopy(small_config)
    # Keep the warmup on the random-action branch so the test stays fast; this
    # test validates real env interaction and relevance marking, not inference.
    config.contextLength = 2000
    # Large enough to retain all four ~1000-step episodes without ring overwrite.
    config.buffer.capacity = 20000
    dreamer = DreamerV4(
        (3, 64, 64), 3, [-1, -1, -1], [1, 1, 1], torch.device("cpu"), config
    )
    try:
        dreamer.environmentInteraction(
            env, num_episodes=4, seed=1, relevant_episode_quantile=0.75
        )
    finally:
        env.close()

    n = len(dreamer.buffer)
    relevant = dreamer.buffer.relevant[:n].reshape(-1)
    dones = dreamer.buffer.dones[:n].reshape(-1)
    assert dreamer.total_episodes == 4
    assert n >= 4
    assert relevant.any() and (~relevant).any(), "both relevance groups must be non-empty"

    start = 0
    labels = []
    for index in range(n):
        if dones[index] > 0.5:
            assert len(set(relevant[start:index + 1].tolist())) == 1, (
                "relevance must cover the whole episode"
            )
            labels.append(bool(relevant[start]))
            start = index + 1
    assert start == n, "every real episode must end with done=1"
    assert len(set(labels)) == 2, "at least one relevant and one uniform episode"


def test_phase2_step(small_dreamer, small_batch):
    metrics = small_dreamer.trainPhase2(small_batch)
    assert "bcLoss" in metrics
    assert "rewardLoss" in metrics


def test_phase3_requires_frozen_prior(small_dreamer, small_batch):
    with pytest.raises(RuntimeError):
        small_dreamer.trainPhase3(small_batch)


def test_phase2_does_not_set_frozen_prior_each_step(small_dreamer, small_batch):
    small_dreamer.trainPhase2(small_batch)
    assert small_dreamer.frozenPrior is None


def test_finalize_phase2_sets_frozen_prior(small_dreamer, small_batch):
    small_dreamer.trainPhase2(small_batch)
    small_dreamer.finalizePhase2()
    assert small_dreamer.frozenPrior is not None
    assert not any(p.requires_grad for p in small_dreamer.frozenPrior.parameters())
    assert not small_dreamer.frozenPrior.training


def test_phase3_uses_imagine_latent(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    call_count = [0]
    original_imagine = small_dreamer.dynamics.imagineLatent

    def counting_imagine(*args, **kwargs):
        call_count[0] += 1
        return original_imagine(*args, **kwargs)

    monkeypatch.setattr(small_dreamer.dynamics, "imagineLatent", counting_imagine)
    small_dreamer.trainPhase3(small_batch)
    assert call_count[0] == small_dreamer.imagination_horizon


def test_phase3_logp_from_sampled_actions(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()

    def fixed_sample(self, sample_shape=torch.Size()):
        shape = self.batch_shape + self.event_shape
        return torch.zeros(shape, dtype=torch.long, device=self.base_dist.logits.device)

    monkeypatch.setattr(Independent, "sample", fixed_sample)
    captured = {}

    def fake_pmpo(logp, policy_dist, prior_dist, advantages, alpha=0.5, beta=0.3):
        captured["logp"] = logp.detach().clone()
        action_size = policy_dist.event_shape[0]
        expected = policy_dist.log_prob(
            torch.zeros(logp.shape[0], logp.shape[1], action_size, dtype=torch.long, device=logp.device)
        )
        assert torch.allclose(logp, expected, atol=1e-6)
        return original_pmpo_loss(logp, policy_dist, prior_dist, advantages, alpha=alpha, beta=beta)

    monkeypatch.setattr(dreamer_mod, "pmpo_loss", fake_pmpo)
    small_dreamer.trainPhase3(small_batch)
    assert "logp" in captured


def test_phase3_dynamics_frozen(small_dreamer, small_batch):
    small_dreamer.freezePolicyPrior()
    before = {name: p.detach().clone() for name, p in small_dreamer.dynamics.named_parameters()}
    small_dreamer.trainPhase3(small_batch)
    for name, p in small_dreamer.dynamics.named_parameters():
        assert torch.allclose(p, before[name], atol=1e-6)


def test_phase3_advantages_from_imagination(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    captured = {}

    def fake_pmpo(logp, policy_dist, prior_dist, advantages, alpha=0.5, beta=0.3):
        captured["advantages"] = advantages.detach().clone()
        return original_pmpo_loss(logp, policy_dist, prior_dist, advantages, alpha=alpha, beta=beta)

    monkeypatch.setattr(dreamer_mod, "pmpo_loss", fake_pmpo)
    small_dreamer.trainPhase3(small_batch)
    assert captured["advantages"].shape == (
        small_batch.observations.shape[0],
        small_dreamer.imagination_horizon - 1,
    )
    assert torch.isfinite(captured["advantages"]).all()


class RecordingPolicyDist:
    def __init__(self, distance):
        self.distance = distance
        self.targets = []

    def log_prob(self, value):
        self.targets.append(value.detach().clone())
        return torch.zeros(value.shape[:-1], device=value.device, dtype=torch.float32)


def test_phase2_mtp_distance0_predicts_current(small_dreamer, small_batch, monkeypatch):
    records = {d: [] for d in range(small_dreamer.mtp_length + 1)}

    def fake_policy_forward(h, distance=0):
        dist = RecordingPolicyDist(distance)
        records[distance].append((h.shape[1], dist))
        return dist

    monkeypatch.setattr(small_dreamer.policyHead, "forward", fake_policy_forward)
    small_dreamer.trainPhase2(small_batch)

    actions = small_batch.actions
    for distance in range(small_dreamer.mtp_length + 1):
        assert len(records[distance]) == 1
        h_len, dist = records[distance][0]
        assert h_len == actions.shape[1] - distance
        expected_target = discretize_actions(
            actions[:, distance:],
            small_dreamer.action_low,
            small_dreamer.action_high,
            small_dreamer.action_bins,
        )
        assert len(dist.targets) == 1
        assert torch.equal(dist.targets[0], expected_target)


def test_phase3_step(small_dreamer, small_batch):
    small_dreamer.freezePolicyPrior()
    metrics = small_dreamer.trainPhase3(small_batch)
    assert "pmpoloss" in metrics
    assert "valueLoss" in metrics
    assert "advantages" in metrics
    assert torch.isfinite(torch.tensor(metrics["phase3Loss"]))


def test_phase3_only_updates_policy_value(small_dreamer, small_batch):
    small_dreamer.freezePolicyPrior()
    before_tokenizer = copy.deepcopy(small_dreamer.tokenizer.state_dict())
    before_dynamics = copy.deepcopy(small_dreamer.dynamics.state_dict())
    small_dreamer.trainPhase3(small_batch)

    for key, tensor in small_dreamer.tokenizer.state_dict().items():
        assert torch.allclose(tensor, before_tokenizer[key], atol=1e-6)
    for key, tensor in small_dreamer.dynamics.state_dict().items():
        assert torch.allclose(tensor, before_dynamics[key], atol=1e-6)


def _with_relevance(batch, is_relevant):
    out = copy.deepcopy(batch)
    out.isRelevant = is_relevant
    return out


def test_phase1b_raises_on_all_relevant(small_dreamer, small_batch):
    all_relevant = _with_relevance(small_batch, torch.ones_like(small_batch.isRelevant))
    with pytest.raises(ValueError):
        small_dreamer.trainPhase1b(all_relevant)


def test_phase2_raises_on_all_uniform(small_dreamer, small_batch):
    all_uniform = _with_relevance(small_batch, torch.zeros_like(small_batch.isRelevant))
    with pytest.raises(ValueError):
        small_dreamer.trainPhase2(all_uniform)


def test_phase1b_mixed_batch_no_nan(small_dreamer, small_batch):
    metrics = small_dreamer.trainPhase1b(small_batch)
    assert torch.isfinite(torch.tensor(metrics["flowLoss"]))


def test_phase2_mixed_relevance_short_length_full_step(small_dreamer):
    """One real Phase 2 step on a formal-length (T=16) mixed-relevance batch.

    Verifies the whole update chain on CPU in fp32: both data groups are
    non-empty, every loss is finite, the internal backward produces finite
    gradients, ``optimizer.step`` succeeds and the updated parameters stay
    finite.  This deliberately uses the formal short sequence length (16).
    """
    assert torch.is_autocast_enabled() is False
    B, T = 2, 16
    is_relevant = torch.zeros(B, T, 1, dtype=torch.bool)
    is_relevant[:, : T // 2] = True
    batch = AttriDict(
        {
            "observations": torch.rand(B, T, 3, 64, 64),
            "nextObservations": torch.rand(B, T, 3, 64, 64),
            "actions": torch.randn(B, T, 3),
            "rewards": torch.randn(B, T, 1),
            "dones": torch.zeros(B, T, 1),
            "isRelevant": is_relevant,
        }
    )
    for name in ("observations", "nextObservations", "actions", "rewards", "dones"):
        assert getattr(batch, name).dtype == torch.float32

    uniform = small_dreamer._relevance_mask(batch.isRelevant, want_relevant=False)
    relevant = small_dreamer._relevance_mask(batch.isRelevant, want_relevant=True)
    assert uniform.sum().item() > 0
    assert relevant.sum().item() > 0

    metrics = small_dreamer.trainPhase2(batch)
    for key in ("flowLoss", "bcLoss", "rewardLoss", "phase2Loss"):
        assert torch.isfinite(torch.tensor(metrics[key])), f"{key} not finite"

    phase2_params = (
        list(small_dreamer.dynamics.parameters())
        + list(small_dreamer.policyHead.parameters())
        + list(small_dreamer.rewardHead.parameters())
    )
    grad_count = 0
    for parameter in phase2_params:
        if parameter.grad is not None:
            grad_count += 1
            assert torch.isfinite(parameter.grad).all(), "gradient became NaN/Inf"
    assert grad_count > 0, "no gradients were produced by the Phase 2 backward"

    dtypes = {p.dtype for p in small_dreamer.parameters()}
    assert dtypes == {torch.float32}
    for parameter in small_dreamer.parameters():
        assert torch.isfinite(parameter).all(), "parameter became NaN/Inf"
    assert not torch.is_autocast_enabled()


def test_phase2_mixed_batch_no_nan(small_dreamer, small_batch):
    metrics = small_dreamer.trainPhase2(small_batch)
    for key in ("bcLoss", "rewardLoss", "phase2Loss", "flowLoss"):
        assert torch.isfinite(torch.tensor(metrics[key]))


def test_relevance_mask_shape_handling(small_dreamer):
    B, T = 2, 8
    is_relevant = torch.zeros(B, T, 1, dtype=torch.bool)
    is_relevant[:, :4] = True
    uniform = small_dreamer._relevance_mask(is_relevant, want_relevant=False)
    relevant = small_dreamer._relevance_mask(is_relevant, want_relevant=True)
    assert uniform.shape == (B, T)
    assert relevant.shape == (B, T)
    assert not uniform[:, :4].any()
    assert (uniform[:, 4:]).all()
    assert relevant[:, :4].all()
    assert not relevant[:, 4:].any()


def test_masked_mean_empty_group_raises(small_dreamer):
    per_sample = torch.rand(2, 8)
    empty = torch.zeros(2, 8, dtype=torch.bool)
    with pytest.raises(ValueError):
        small_dreamer._masked_mean(per_sample, empty, context="Phase2 BC (relevant)")


def test_masked_mean_selects_correct_subset(small_dreamer):
    per_sample = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :2] = True
    result = small_dreamer._masked_mean(per_sample, mask, context="test (relevant)")
    expected = per_sample[mask].mean()
    assert torch.allclose(result, expected)


def test_masked_mean_ignores_masked_out_values(small_dreamer):
    per_sample = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :2] = True
    base = small_dreamer._masked_mean(per_sample, mask, context="test (relevant)")
    per_sample[0, 7] = 9999.0  # masked-out (uniform) element
    assert torch.allclose(base, small_dreamer._masked_mean(per_sample, mask, context="test (relevant)"))
    per_sample[0, 0] = 9999.0  # included (relevant) element
    assert not torch.allclose(base, small_dreamer._masked_mean(per_sample, mask, context="test (relevant)"))


def _mask_probe(small_dreamer, monkeypatch):
    """Neutralize heavy forwards and record every mask passed to _masked_mean."""
    B, T = 2, 8
    captured = []
    original = small_dreamer._masked_mean

    def fake_masked_mean(per_sample, mask, context):
        captured.append((context, mask.detach().clone()))
        return per_sample.sum()

    monkeypatch.setattr(small_dreamer, "_masked_mean", fake_masked_mean)
    monkeypatch.setattr(
        small_dreamer.dynamics,
        "shortcutForcingLoss",
        lambda z1, actions: (
            torch.zeros(B, T, device=z1.device, requires_grad=True),
            {"flowLoss": 0.0, "flowRegression": 0.0, "flowBootstrap": 0.0},
        ),
    )
    monkeypatch.setattr(
        small_dreamer.dynamics,
        "agentOutputs",
        lambda z, actions, context_z=None: torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        ),
    )
    return captured


def test_phase1b_dynamics_mask_is_uniform(small_dreamer, small_batch, monkeypatch):
    captured = _mask_probe(small_dreamer, monkeypatch)
    small_dreamer.trainPhase1b(small_batch)
    assert len(captured) == 1
    context, mask = captured[0]
    assert "uniform" in context
    assert torch.equal(mask, ~small_batch.isRelevant[..., 0].bool())


def test_phase2_dynamics_mask_is_uniform(small_dreamer, small_batch, monkeypatch):
    captured = _mask_probe(small_dreamer, monkeypatch)
    small_dreamer.trainPhase2(small_batch)
    context, mask = captured[0]
    assert context == "Phase2 dynamics (uniform)"
    assert torch.equal(mask, ~small_batch.isRelevant[..., 0].bool())


def test_phase2_bc_reward_masks_are_relevant(small_dreamer, small_batch, monkeypatch):
    captured = _mask_probe(small_dreamer, monkeypatch)
    small_dreamer.trainPhase2(small_batch)
    # captured[0] = dynamics uniform; [1] = BC relevant; [2] = reward relevant.
    assert len(captured) == 3
    rel = small_batch.isRelevant
    context0, mask0 = captured[0]
    assert context0 == "Phase2 dynamics (uniform)"
    assert torch.equal(mask0, ~rel[..., 0].bool())
    total_relevant = sum(
        rel[:, d:, 0].sum().item() for d in range(small_dreamer.mtp_length + 1)
    )
    for i in (1, 2):
        context, mask = captured[i]
        assert "relevant" in context
        assert mask.sum().item() == total_relevant


def test_phase2_raises_on_all_relevant(small_dreamer, small_batch):
    all_relevant = _with_relevance(small_batch, torch.ones_like(small_batch.isRelevant))
    with pytest.raises(ValueError):
        small_dreamer.trainPhase2(all_relevant)


def test_phase2_mtp_mask_shift_alignment(small_dreamer, small_batch, monkeypatch):
    rel = small_batch.isRelevant  # (B, T, 1)
    B, T, _ = rel.shape
    captured = []
    original = small_dreamer._relevance_mask

    def fake_mask(is_relevant, want_relevant):
        captured.append(is_relevant.detach().clone())
        return original(is_relevant, want_relevant)

    monkeypatch.setattr(small_dreamer, "_relevance_mask", fake_mask)
    monkeypatch.setattr(
        small_dreamer.dynamics,
        "shortcutForcingLoss",
        lambda z1, actions: (
            torch.zeros(B, T, device=z1.device, requires_grad=True),
            {"flowLoss": 0.0, "flowRegression": 0.0, "flowBootstrap": 0.0},
        ),
    )
    monkeypatch.setattr(
        small_dreamer.dynamics,
        "agentOutputs",
        lambda z, actions, context_z=None: torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        ),
    )

    small_dreamer.trainPhase2(small_batch)

    n = small_dreamer.mtp_length + 1
    # One call for the dynamics (uniform) term, then one per MTP distance
    # (the same shifted mask is shared by the BC and reward heads).
    assert len(captured) == 1 + n
    assert torch.equal(captured[0][..., 0], rel[..., 0].bool())
    for dist in range(n):
        expected = rel[:, dist:, 0].bool()
        assert torch.equal(captured[1 + dist][..., 0], expected), f"distance {dist} mask misaligned"


def test_phase2_updates_dynamics_policy_reward(small_dreamer, small_batch):
    before_dynamics = {name: p.detach().clone() for name, p in small_dreamer.dynamics.named_parameters()}
    before_policy = {name: p.detach().clone() for name, p in small_dreamer.policyHead.named_parameters()}
    before_reward = {name: p.detach().clone() for name, p in small_dreamer.rewardHead.named_parameters()}
    small_dreamer.trainPhase2(small_batch)
    for name, p in small_dreamer.dynamics.named_parameters():
        assert not torch.allclose(p, before_dynamics[name], atol=1e-6), f"dynamics {name} unchanged"
    for name, p in small_dreamer.policyHead.named_parameters():
        assert not torch.allclose(p, before_policy[name], atol=1e-6), f"policy {name} unchanged"
    for name, p in small_dreamer.rewardHead.named_parameters():
        assert not torch.allclose(p, before_reward[name], atol=1e-6), f"reward {name} unchanged"


def test_phase2_freezes_tokenizer_and_value(small_dreamer, small_batch):
    before_tokenizer = copy.deepcopy(small_dreamer.tokenizer.state_dict())
    before_value = copy.deepcopy(small_dreamer.valueHead.state_dict())
    small_dreamer.trainPhase2(small_batch)
    for key, tensor in small_dreamer.tokenizer.state_dict().items():
        assert torch.allclose(tensor, before_tokenizer[key], atol=1e-6)
    for key, tensor in small_dreamer.valueHead.state_dict().items():
        assert torch.allclose(tensor, before_value[key], atol=1e-6)


def _fill_per_episode_labeled(small_dreamer, relevant_episodes, uniform_episodes, episode_length):
    for episode in range(relevant_episodes + uniform_episodes):
        relevant = episode < relevant_episodes
        for step in range(episode_length):
            small_dreamer.buffer.add(
                np.zeros((3, 64, 64), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                0.0,
                np.zeros((3, 64, 64), dtype=np.float32),
                step == episode_length - 1,
                relevant=relevant,
            )


def test_phase2_balanced_sampling_never_hits_empty_group(small_dreamer):
    _fill_per_episode_labeled(small_dreamer, relevant_episodes=2, uniform_episodes=2, episode_length=16)
    batch_size = 2
    sequence_size = 6

    for _ in range(200):
        batch = small_dreamer.buffer.sample(
            batch_size, sequence_size, balance_relevance=True
        )
        assert bool(batch.isRelevant.any())
        assert bool((~batch.isRelevant).any())

    for _ in range(3):
        batch = small_dreamer.buffer.sample(
            batch_size, sequence_size, balance_relevance=True
        )
        metrics = small_dreamer.trainPhase2(batch)
        for key in ("flowLoss", "bcLoss", "rewardLoss", "phase2Loss"):
            assert torch.isfinite(torch.tensor(metrics[key]))
    dtypes = {p.dtype for p in small_dreamer.parameters()}
    assert dtypes == {torch.float32}


def test_phase2_unbalanced_one_class_batch_raises(small_dreamer, small_batch):
    # The fail-fast empty-group check must stay in place (not silently masked).
    all_relevant = copy.deepcopy(small_batch)
    all_relevant.isRelevant = torch.ones_like(small_batch.isRelevant)
    with pytest.raises(ValueError):
        small_dreamer.trainPhase2(all_relevant)
