import pytest
import torch

from dynamics import (
    DynamicsTransformer,
    build_legal_tau_d_pairs,
    sample_tau_d_from_pairs,
    validate_step_bins,
)


def make_small_dynamics(small_config):
    return DynamicsTransformer(
        small_config.tokenizer.latentTokens,
        small_config.tokenizer.latentDim,
        3,
        small_config.dynamics,
    )


def test_forward_shape(small_config):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z = torch.randn(B, T, N, D)
    tau_idx = torch.zeros(B, T, dtype=torch.long)
    d_idx = torch.zeros(B, T, dtype=torch.long)
    actions = torch.randn(B, T, 3)
    out = dyn(z, tau_idx, d_idx, actions)
    assert out.shape == z.shape


def test_agent_outputs_shape(small_config):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 3)
    h = dyn.agentOutputs(z, actions)
    assert h.shape == (B, T, small_config.dynamics.modelDim)


def test_shortcut_forcing_loss_shape(small_config):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z1 = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 3)
    per_sample, metrics = dyn.shortcutForcingLoss(z1, actions)
    assert per_sample.shape == (B, T)
    per_sample.mean().backward()
    assert "flowLoss" in metrics
    assert torch.isfinite(torch.tensor(metrics["flowLoss"]))
    assert torch.isfinite(torch.tensor(metrics["flowBootstrap"]))


def test_shortcut_forcing_excludes_tau1_d_gt_0(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 2, 8, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z1 = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 3)
    captured = []

    def fake_forward(z_tilde, tau_idx, d_idx, actions, agent_tokens=None, context_z=None, **kwargs):
        captured.append((tau_idx.detach().clone(), d_idx.detach().clone()))
        return z_tilde

    monkeypatch.setattr(dyn, "forward", fake_forward)
    dyn.shortcutForcingLoss(z1, actions)
    assert captured
    tau_idx, d_idx = captured[0]
    assert not ((tau_idx == dyn.tau_bins - 1) & (d_idx > 0)).any()
    assert (tau_idx != dyn.tau_bins - 1).all()


def test_build_legal_tau_d_pairs_within_bounds(small_config):
    tau_values = [i / (small_config.dynamics.tauBins - 1) for i in range(small_config.dynamics.tauBins)]
    step_bins = list(small_config.dynamics.stepBins)
    pairs = build_legal_tau_d_pairs(tau_values, step_bins)
    assert pairs
    for tau_idx, d_idx in pairs:
        assert tau_values[tau_idx] + step_bins[d_idx] <= 1.0 + 1e-9
        # Half-step teacher must also stay in bounds.
        assert tau_values[tau_idx] + step_bins[d_idx] / 2.0 <= 1.0 + 1e-9


def test_build_legal_tau_d_pairs_excludes_tau1(small_config):
    tau_values = [i / (small_config.dynamics.tauBins - 1) for i in range(small_config.dynamics.tauBins)]
    step_bins = list(small_config.dynamics.stepBins)
    pairs = build_legal_tau_d_pairs(tau_values, step_bins)
    tau1_idx = len(tau_values) - 1
    assert not any(tau_idx == tau1_idx for tau_idx, _ in pairs)


def test_sample_tau_d_respects_grid(small_config):
    tau_values = [i / (small_config.dynamics.tauBins - 1) for i in range(small_config.dynamics.tauBins)]
    step_bins = list(small_config.dynamics.stepBins)
    pairs = build_legal_tau_d_pairs(tau_values, step_bins)
    tau_idx, d_idx = sample_tau_d_from_pairs(pairs, (50, 40), torch.device("cpu"))
    assert tau_idx.shape == (50, 40)
    assert d_idx.shape == (50, 40)
    for i in range(50):
        for j in range(40):
            tau = tau_values[int(tau_idx[i, j])]
            d = step_bins[int(d_idx[i, j])]
            assert tau + d <= 1.0 + 1e-9
            assert tau + d / 2.0 <= 1.0 + 1e-9


def test_validate_step_bins_legal():
    assert validate_step_bins([0.25, 0.5, 1.0]) == [0.25, 0.5, 1.0]
    assert validate_step_bins([0.25]) == [0.25]
    assert validate_step_bins([0.125, 0.25, 0.5, 1.0]) == [0.125, 0.25, 0.5, 1.0]


def test_validate_step_bins_illegal():
    with pytest.raises(ValueError):
        validate_step_bins([0.3, 0.7, 1.0])
    with pytest.raises(ValueError):
        validate_step_bins([1.0, 0.5, 0.25])  # not increasing
    with pytest.raises(ValueError):
        validate_step_bins([])
    with pytest.raises(ValueError):
        validate_step_bins([0.25, 0.5, 0.5])


def test_dynamics_rejects_invalid_step_bins(small_config):
    import copy

    bad = copy.deepcopy(small_config.dynamics)
    bad["stepBins"] = [0.3, 0.7, 1.0]
    with pytest.raises(ValueError):
        DynamicsTransformer(
            small_config.tokenizer.latentTokens,
            small_config.tokenizer.latentDim,
            3,
            bad,
        )


def test_shortcut_forcing_bootstrap_factor(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 1, 2, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z1 = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 3)

    tau_idx = torch.full((B, T), 2, dtype=torch.long)  # tau = 0.25
    d_idx = torch.full((B, T), 1, dtype=torch.long)    # d = 0.5 (non-min)
    monkeypatch.setattr(dyn, "_sample_tau_d", lambda B, T, device: (tau_idx, d_idx))

    def fake_forward(z_tilde, tau_idx_, d_idx_, actions_, agent_tokens=None, context_z=None, **kwargs):
        tau = dyn._tau_value(tau_idx_)
        return z_tilde + (1.0 - tau)[..., None, None] * d_idx_.float()[..., None, None]

    monkeypatch.setattr(dyn, "forward", fake_forward)

    per_sample, metrics = dyn.shortcutForcingLoss(z1, actions)
    tau = dyn._tau_value(tau_idx)
    expected = (0.9 * tau + 0.1) * (1.0 - tau) ** 2
    assert torch.allclose(per_sample, expected, atol=1e-6)


def test_shortcut_forcing_min_step_uses_regression(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 1, 2, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z1 = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 3)

    tau_idx = torch.full((B, T), 2, dtype=torch.long)  # tau = 0.25
    d_idx = torch.zeros((B, T), dtype=torch.long)      # min step
    monkeypatch.setattr(dyn, "_sample_tau_d", lambda B, T, device: (tau_idx, d_idx))

    def fake_forward(z_tilde, tau_idx_, d_idx_, actions_, agent_tokens=None, context_z=None, **kwargs):
        return z_tilde

    monkeypatch.setattr(dyn, "forward", fake_forward)

    per_sample, metrics = dyn.shortcutForcingLoss(z1, actions)
    assert torch.isfinite(per_sample).all()
    assert torch.isfinite(torch.tensor(metrics["flowRegression"]))
    # All samples are at the minimum step, so no bootstrap target exists.
    assert metrics["flowBootstrap"] == 0.0


def test_imagine_latent_output_shape(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, C, N, D = 2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    ctx_z = torch.randn(B, C, N, D)
    actions = torch.randn(B, 1, 3)

    def fake_forward(z_tilde, tau_idx, d_idx, actions, agent_tokens=None, context_z=None, **kwargs):
        return z_tilde

    monkeypatch.setattr(dyn, "forward", fake_forward)
    context_actions = torch.randn(B, C, 3)
    z = dyn.imagineLatent(ctx_z, actions, K=4, context_actions=context_actions)
    assert z.shape == (B, 1, N, D)


def test_imagine_latent_tau_progression(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, C, N, D = 2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    ctx_z = torch.randn(B, C, N, D)
    actions = torch.randn(B, 1, 3)
    taus = []

    def fake_tau_index(tau):
        taus.append(tau.detach().clone())
        return torch.zeros_like(tau, dtype=torch.long)

    def fake_forward(z_tilde, tau_idx, d_idx, actions, agent_tokens=None, context_z=None, **kwargs):
        return z_tilde

    monkeypatch.setattr(dyn, "_tau_index", fake_tau_index)
    monkeypatch.setattr(dyn, "forward", fake_forward)
    dyn.imagineLatent(ctx_z, actions, K=4, context_actions=torch.randn(B, C, 3))
    assert len(taus) == 4
    expected = [0.0, 0.25, 0.5, 0.75]
    for actual, ref in zip(taus, expected):
        assert torch.allclose(actual, torch.full_like(actual, ref), atol=1e-6)


def test_ctx_tau_is_legal_and_trained(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    assert dyn.tau_bins == 9
    # The context corruption signal must land on the legal/trained shortcut
    # grid: the largest trained signal for the minimum step is 1 - d_min = 0.75
    # (index 6), not the clean 1.0 (index 8) and not the off-grid 0.875 (7).
    expected_idx = dyn.contextSignalIndex(tau_ctx=0.1)
    assert expected_idx == 6
    assert expected_idx != dyn.tau_bins - 1
    assert dyn.tau_values[expected_idx] + dyn.step_bins[0] <= 1.0 + 1e-9

    B, T, N, D = 1, 1, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z = torch.randn(B, T, N, D)
    context_z = torch.randn(B, 2, N, D)
    actions = torch.randn(B, T, 3)
    frame_calls = []
    original_frame = dyn._frame_tokens

    def capture_frame(z_, tau_idx, d_idx, actions_, agent_tokens=None):
        frame_calls.append(tau_idx.detach().clone())
        return original_frame(z_, tau_idx, d_idx, actions_, agent_tokens)

    monkeypatch.setattr(dyn, "_frame_tokens", capture_frame)
    dyn._forward_tokens(z, torch.zeros_like(z[:, :, 0, 0], dtype=torch.long),
                        torch.zeros_like(z[:, :, 0, 0], dtype=torch.long), actions,
                        context_z=context_z, context_actions=torch.randn(B, 2, 3))
    assert len(frame_calls) == 2
    ctx_tau_idx = frame_calls[1]
    assert ctx_tau_idx.unique().tolist() == [expected_idx]
    assert ctx_tau_idx.unique().item() != dyn.tau_bins - 1


def test_imagine_latent_context_corrupted(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, C, N, D = 2, 4, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    ctx_z = torch.randn(B, C, N, D)
    actions = torch.randn(B, 1, 3)
    captured = []

    def fake_forward(z_tilde, tau_idx, d_idx, actions, agent_tokens=None, context_z=None, **kwargs):
        captured.append(context_z.detach().clone())
        return z_tilde

    monkeypatch.setattr(dyn, "forward", fake_forward)
    dyn.imagineLatent(ctx_z, actions, K=2, context_actions=torch.randn(B, C, 3))
    assert captured
    context = captured[0]
    assert context.shape == ctx_z.shape
    assert not torch.allclose(context, ctx_z)


def test_forward_tokens_context_actions_dtype_mismatch_raises(small_config):
    dyn = make_small_dynamics(small_config)
    B, C, N, D = 1, 2, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z = torch.randn(B, 1, N, D)  # float32
    context_z = torch.randn(B, C, N, D)
    actions = torch.randn(B, 1, 3)
    tau_idx = torch.zeros(B, 1, dtype=torch.long)
    d_idx = torch.zeros(B, 1, dtype=torch.long)
    bad = torch.randn(B, C, 3, dtype=torch.float64)
    with pytest.raises(ValueError) as excinfo:
        dyn._forward_tokens(
            z, tau_idx, d_idx, actions, context_z=context_z, context_actions=bad
        )
    assert "dtype" in str(excinfo.value)


def test_forward_tokens_context_actions_device_mismatch_raises(small_config):
    dyn = make_small_dynamics(small_config)
    B, C, N, D = 1, 2, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z = torch.randn(B, 1, N, D)  # cpu
    context_z = torch.randn(B, C, N, D)
    actions = torch.randn(B, 1, 3)
    tau_idx = torch.zeros(B, 1, dtype=torch.long)
    d_idx = torch.zeros(B, 1, dtype=torch.long)
    if torch.cuda.is_available():
        other = torch.device("cuda")
    else:
        try:
            torch.empty(0, device="meta")
        except Exception:
            pytest.skip("no non-CPU device available for a device-mismatch test")
        other = torch.device("meta")  # device-mismatch check only, no arithmetic
    bad = torch.empty(B, C, 3, device=other)
    with pytest.raises(ValueError) as excinfo:
        dyn._forward_tokens(
            z, tau_idx, d_idx, actions, context_z=context_z, context_actions=bad
        )
    assert "device" in str(excinfo.value)


def test_shortcut_forcing_rejects_invalid_tau_d(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 1, 2, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z1 = torch.randn(B, T, N, D)
    actions = torch.randn(B, T, 3)

    # Force an illegal (tau=1, d>0) sample that violates tau + d <= 1.
    tau_idx = torch.full((B, T), dyn.tau_bins - 1, dtype=torch.long)
    d_idx = torch.ones((B, T), dtype=torch.long)
    monkeypatch.setattr(dyn, "_sample_tau_d", lambda B, T, device: (tau_idx, d_idx))

    def fake_forward(z_tilde, tau_idx_, d_idx_, actions_, agent_tokens=None, context_z=None, **kwargs):
        return z_tilde

    monkeypatch.setattr(dyn, "forward", fake_forward)
    with pytest.raises(RuntimeError):
        dyn.shortcutForcingLoss(z1, actions)


def test_sample_tau_d_conditional_grid(small_config):
    dyn = make_small_dynamics(small_config)
    tau_idx, d_idx = dyn._sample_tau_d(100, 100, torch.device("cpu"))
    tau_values = torch.tensor(dyn.tau_values)
    step_bins = torch.tensor(dyn.step_bins)
    tau = tau_values[tau_idx]
    d = step_bins[d_idx]
    assert (tau + d <= 1.0 + 1e-6).all()
    assert (tau + d / 2.0 <= 1.0 + 1e-6).all()
    assert (tau < 1.0).all()
    # tau must lie on the step-aligned grid {0, d, 2d, ...}: tau / d is integer.
    assert torch.allclose(tau % d, torch.zeros_like(tau), atol=1e-6)


def test_sample_tau_d_reaches_full_range(small_config):
    dyn = make_small_dynamics(small_config)
    tau_values = torch.tensor(dyn.tau_values)
    step_bins = torch.tensor(dyn.step_bins)
    tau_idx, d_idx = dyn._sample_tau_d(2000, 1, torch.device("cpu"))
    tau = tau_values[tau_idx]
    d = step_bins[d_idx]
    min_d = dyn.step_bins[0]
    min_mask = d == min_d
    K_min = int(round(1.0 / min_d))
    reached = set((tau[min_mask] / min_d).round().long().tolist())
    assert reached == set(range(K_min))


def test_sample_tau_d_max_step_tau_zero(small_config):
    dyn = make_small_dynamics(small_config)
    tau_idx, d_idx = dyn._sample_tau_d(200, 1, torch.device("cpu"))
    max_d_idx = len(dyn.step_bins) - 1
    assert (d_idx == max_d_idx).any()
    assert (tau_idx[d_idx == max_d_idx] == 0).all()


def test_shortcut_forcing_teacher_no_grad_student_grad(small_config, monkeypatch):
    dyn = make_small_dynamics(small_config)
    B, T, N, D = 1, 2, small_config.tokenizer.latentTokens, small_config.tokenizer.latentDim
    z1 = torch.randn(B, T, N, D, requires_grad=True)
    actions = torch.randn(B, T, 3)
    tau_idx = torch.full((B, T), 2, dtype=torch.long)  # tau = 0.25
    d_idx = torch.full((B, T), 1, dtype=torch.long)    # d = 0.5 (non-min -> teacher)
    monkeypatch.setattr(dyn, "_sample_tau_d", lambda B, T, device: (tau_idx, d_idx))

    calls = []

    def fake_forward(z_tilde, tau_idx_, d_idx_, actions_, agent_tokens=None, context_z=None, **kwargs):
        out = z_tilde * 2.0
        calls.append(out.requires_grad)
        return out

    monkeypatch.setattr(dyn, "forward", fake_forward)
    per_sample, _ = dyn.shortcutForcingLoss(z1, actions)
    assert per_sample.isfinite().all()
    # First call is the student (grad-enabled); the two half-steps are no-grad.
    assert calls[0] is True
    assert calls[1] is False
    assert calls[2] is False
