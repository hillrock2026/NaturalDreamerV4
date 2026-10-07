"""Task 8 regression tests: Phase 3 imagination time semantics.

These tests pin the project-wide action convention to **arriving**:

    slot[t] = a[t-1]        (the action that produced frame/state t)
    slot[0] = 0             (dummy; reference behaviour)

Training, Phase 2, Phase 3 context/imagination and the online history are all
required to use this single convention.  The tests deliberately use unique
per-step codes rather than shape checks so that an off-by-one cannot pass.

They also pin:
  * context actions are the real batch actions (not zeros),
  * agentOutputs and imagineLatent share the same context tensor,
  * reward/value/lambda-return indices,
  * Phase 3 dropout/eval handling,
  * gradient boundaries,
  * read-only compatibility with the published 320k checkpoint.
"""

import os
import copy

import pytest
import torch
from torch.distributions import Independent

from dreamer import DreamerV4
from heads import discrete_to_actions
from utils import computeLambdaValues

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "CarRacing-v3_V4RunOffline_320k_final.pth",
)


def _unique_actions(B, T, A=3):
    """Action where every (batch, time) has a distinct value."""
    base = torch.arange(1, B * T + 1, dtype=torch.float32).reshape(B, T, 1)
    return base.expand(B, T, A).clone()


# ---------------------------------------------------------------------------
# 1. Action convention timeline
# ---------------------------------------------------------------------------

def test_arriving_actions_helper():
    actions = _unique_actions(2, 4)
    arriving = DreamerV4._arrivingActions(actions)
    assert torch.equal(arriving[:, 0], torch.zeros(2, 3))
    assert torch.equal(arriving[:, 1:], actions[:, :-1])


def test_phase1b_feeds_arriving_actions(small_dreamer, small_batch, monkeypatch):
    captured = {}

    def fake_shortcut(z1, actions):
        captured["actions"] = actions.detach().clone()
        B, T = z1.shape[:2]
        return (
            torch.zeros(B, T, device=z1.device, requires_grad=True),
            {"flowLoss": 0.0, "flowRegression": 0.0, "flowBootstrap": 0.0},
        )

    monkeypatch.setattr(small_dreamer.dynamics, "shortcutForcingLoss", fake_shortcut)
    small_dreamer.trainPhase1b(small_batch)

    expected = DreamerV4._arrivingActions(small_batch.actions)
    assert torch.equal(captured["actions"], expected)
    # The first slot must be the dummy, never a real action.
    assert torch.equal(captured["actions"][:, 0], torch.zeros_like(captured["actions"][:, 0]))


def test_phase2_feeds_arriving_actions_to_dynamics(small_dreamer, small_batch, monkeypatch):
    captured = {}

    def fake_shortcut(z1, actions):
        captured["flow"] = actions.detach().clone()
        B, T = z1.shape[:2]
        return (
            torch.zeros(B, T, device=z1.device, requires_grad=True),
            {"flowLoss": 0.0, "flowRegression": 0.0, "flowBootstrap": 0.0},
        )

    def fake_agent(z, actions, context_z=None, context_tau_idx=None,
                   context_actions=None, **kwargs):
        captured["agent"] = actions.detach().clone()
        B, T = z.shape[:2]
        return torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        )

    monkeypatch.setattr(small_dreamer.dynamics, "shortcutForcingLoss", fake_shortcut)
    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", fake_agent)
    small_dreamer.trainPhase2(small_batch)

    expected = DreamerV4._arrivingActions(small_batch.actions)
    assert torch.equal(captured["flow"], expected)
    assert torch.equal(captured["agent"], expected)


def test_phase2_mtp_targets_stay_executing(small_dreamer, small_batch, monkeypatch):
    """distance=0 must predict a_t (executing target) from the arriving input."""

    def fake_shortcut(z1, actions):
        B, T = z1.shape[:2]
        return (
            torch.zeros(B, T, device=z1.device, requires_grad=True),
            {"flowLoss": 0.0, "flowRegression": 0.0, "flowBootstrap": 0.0},
        )

    def fake_agent(z, actions, context_z=None, context_tau_idx=None,
                   context_actions=None, **kwargs):
        B, T = z.shape[:2]
        return torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        )

    monkeypatch.setattr(small_dreamer.dynamics, "shortcutForcingLoss", fake_shortcut)
    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", fake_agent)

    recorded = {}

    class Rec:
        def __init__(self, d):
            self.d = d

        def log_prob(self, value):
            recorded.setdefault(self.d, []).append(value.detach().clone())
            return torch.zeros(value.shape[:-1], device=value.device, requires_grad=True)

    monkeypatch.setattr(small_dreamer.policyHead, "forward", lambda h, distance=0: Rec(distance))
    small_dreamer.trainPhase2(small_batch)

    # distance=0 target must be the executing action at the same frame.
    d0 = recorded[0][0]
    assert d0.shape[:2] == (
        small_batch.actions.shape[0],
        small_batch.actions.shape[1],
    )
    assert not torch.equal(
        d0, DreamerV4._arrivingActions(small_batch.actions)
    ), "MTP target must not be the shifted input (that would be leakage)"


# ---------------------------------------------------------------------------
# 2. Context actions
# ---------------------------------------------------------------------------

def test_phase3_context_actions_are_real_arriving(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    captured = {"ctx_actions": None, "current_actions": None}

    def fake_agent(z, actions, context_z=None, context_tau_idx=None,
                   context_actions=None, **kwargs):
        if captured["ctx_actions"] is None:
            captured["ctx_actions"] = (
                None if context_actions is None else context_actions.detach().clone()
            )
            captured["current_actions"] = actions.detach().clone()
        B, T = z.shape[:2]
        return torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        )

    def fake_imagine(ctx_z, actions, K=4, context_tau_idx=None, **kwargs):
        return ctx_z[:, -1:].clone()

    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", fake_agent)
    monkeypatch.setattr(small_dreamer.dynamics, "imagineLatent", fake_imagine)

    small_dreamer.trainPhase3(small_batch)

    C = min(small_dreamer.context_length, small_batch.actions.shape[1])
    arriving = DreamerV4._arrivingActions(small_batch.actions[:, :C])
    assert captured["ctx_actions"] is not None, "context actions must be passed to agentOutputs"
    # context = frames [0, C-2], current = frame C-1; together they are the
    # full arriving context action sequence.
    full = torch.cat([captured["ctx_actions"], captured["current_actions"]], dim=1)
    assert torch.equal(full, arriving)
    # And it must not be the all-zero placeholder.
    assert arriving.abs().sum() > 0
    assert not torch.equal(captured["ctx_actions"], torch.zeros_like(captured["ctx_actions"]))


def test_phase3_agent_and_imagine_share_context(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    seen = {"agent": [], "imagine": []}

    def fake_agent(z, actions, context_z=None, context_tau_idx=None,
                   context_actions=None, **kwargs):
        seen["agent"].append(
            {
                "z": None if context_z is None else context_z.detach().clone(),
                "a": None if context_actions is None else context_actions.detach().clone(),
                "tau": context_tau_idx,
                "current_a": actions.detach().clone(),
            }
        )
        B, T = z.shape[:2]
        return torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        )

    def fake_imagine(ctx_z, actions, K=4, context_tau_idx=None,
                     context_actions=None, **kwargs):
        seen["imagine"].append(
            {
                "z": ctx_z.detach().clone(),
                "a": None if context_actions is None else context_actions.detach().clone(),
                "tau": context_tau_idx,
            }
        )
        return ctx_z[:, -1:].clone()

    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", fake_agent)
    monkeypatch.setattr(small_dreamer.dynamics, "imagineLatent", fake_imagine)

    small_dreamer.trainPhase3(small_batch)

    assert len(seen["agent"]) == len(seen["imagine"]) == small_dreamer.imagination_horizon
    # For every rollout step the agent context is exactly the imagine context
    # minus the current frame -> both read the same corrupted history.
    for agent, imagine in zip(seen["agent"], seen["imagine"]):
        assert torch.equal(agent["z"], imagine["z"][:, :-1])
        # P1-A: the action condition must match too, not only z/tau.
        assert imagine["a"] is not None, "imagineLatent must receive context_actions"
        assert torch.equal(agent["a"], imagine["a"][:, :-1])
        # The agent's full arriving sequence (context + current) is exactly the
        # imagine context action sequence (imagine has one extra context frame).
        assert torch.equal(
            torch.cat([agent["a"], agent["current_a"]], dim=1), imagine["a"]
        )
        assert agent["tau"] == imagine["tau"]
        # No silent all-zero placeholder once the real history exists.
        if imagine["a"].abs().sum() > 0:
            assert not torch.equal(imagine["a"], torch.zeros_like(imagine["a"]))


def test_phase3_imagine_context_actions_are_real_arriving(small_dreamer, monkeypatch):
    """Value-level P1-A test: imagineLatent must receive the real arriving
    context actions (unique codes), never the internal all-zero placeholder."""
    small_dreamer.freezePolicyPrior()

    def _unique_actions(B, T):
        base = torch.arange(1, B * T + 1, dtype=torch.float32).reshape(B, T, 1)
        return base.expand(B, T, small_dreamer.action_size).clone()

    from attridict import AttriDict

    B, T = 1, 6
    batch = AttriDict(
        {
            "observations": torch.rand(B, T, 3, 64, 64),
            "actions": _unique_actions(B, T),
            "rewards": torch.randn(B, T, 1),
            "dones": torch.zeros(B, T, 1),
            "isRelevant": torch.ones(B, T, 1, dtype=torch.bool),
        }
    )
    seen = []

    def fake_agent(z, actions, context_z=None, context_tau_idx=None,
                   context_actions=None, **kwargs):
        B_, T_ = z.shape[:2]
        return torch.zeros(
            B_, T_, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        )

    def fake_imagine(ctx_z, actions, K=4, context_tau_idx=None,
                     context_actions=None, **kwargs):
        seen.append(context_actions.detach().clone())
        return ctx_z[:, -1:].clone()

    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", fake_agent)
    monkeypatch.setattr(small_dreamer.dynamics, "imagineLatent", fake_imagine)

    small_dreamer.trainPhase3(batch)

    C = min(small_dreamer.context_length, T)
    arriving = DreamerV4._arrivingActions(batch.actions[:, :C])
    # Step 0 imagine context == the full arriving history [0, a0, a1, ...].
    step0 = seen[0]
    assert torch.equal(step0, arriving)
    # Unique codes prove the first slot is the only dummy and the rest are a[t-1].
    assert torch.equal(step0[:, 0], torch.zeros_like(step0[:, 0]))
    assert torch.equal(step0[:, 1:], batch.actions[:, : C - 1])
    assert step0.abs().sum() > 0
    assert not torch.equal(step0, torch.zeros_like(step0))


def test_imagine_latent_receives_context_actions_value_level(small_dreamer, monkeypatch):
    """Real ``imagineLatent`` -> ``forward`` pass-through, value-level."""
    dyn = small_dreamer.dynamics
    B, C = 2, 4
    ctx_z = torch.randn(B, C, dyn.latent_tokens, dyn.latent_dim)
    ctx_actions = (
        torch.arange(1, B * C + 1, dtype=torch.float32).reshape(B, C, 1)
        .expand(B, C, dyn.action_size)
        .clone()
    )
    action = torch.randn(B, 1, dyn.action_size)
    captured = []

    def spy_forward(z_tilde, tau_idx, d_idx, actions, agent_tokens=None,
                    context_z=None, context_tau_idx=None, context_actions=None):
        captured.append(
            None if context_actions is None else context_actions.detach().clone()
        )
        return z_tilde

    monkeypatch.setattr(dyn, "forward", spy_forward)
    z = dyn.imagineLatent(ctx_z, action, K=4, context_actions=ctx_actions)

    assert z.shape == (B, 1, dyn.latent_tokens, dyn.latent_dim)
    assert captured and all(c is not None for c in captured)
    for c in captured:
        assert torch.equal(c, ctx_actions)
    assert not torch.equal(captured[0], torch.zeros_like(captured[0]))


def test_forward_tokens_requires_context_actions(small_dreamer):
    dyn = small_dreamer.dynamics
    B, C = 1, 2
    z = torch.randn(B, 1, dyn.latent_tokens, dyn.latent_dim)
    context_z = torch.randn(B, C, dyn.latent_tokens, dyn.latent_dim)
    actions = torch.randn(B, 1, dyn.action_size)
    tau_idx = torch.zeros(B, 1, dtype=torch.long)
    d_idx = torch.zeros(B, 1, dtype=torch.long)
    with pytest.raises(ValueError):
        dyn._forward_tokens(z, tau_idx, d_idx, actions, context_z=context_z)
    with pytest.raises(ValueError):
        dyn.imagineLatent(context_z, actions, K=2)


def test_context_actions_shape_mismatch_raises(small_dreamer):
    dyn = small_dreamer.dynamics
    B, C = 2, 3
    z = torch.randn(B, 1, dyn.latent_tokens, dyn.latent_dim)
    context_z = torch.randn(B, C, dyn.latent_tokens, dyn.latent_dim)
    actions = torch.randn(B, 1, dyn.action_size)
    tau_idx = torch.zeros(B, 1, dtype=torch.long)
    d_idx = torch.zeros(B, 1, dtype=torch.long)
    bad_shape = [
        torch.randn(B, C - 1, dyn.action_size),
        torch.randn(B + 1, C, dyn.action_size),
        torch.randn(B, C, dyn.action_size + 1),
    ]
    for bad in bad_shape:
        with pytest.raises(ValueError):
            dyn._forward_tokens(
                z, tau_idx, d_idx, actions, context_z=context_z, context_actions=bad
            )


def test_empty_context_equals_no_context(small_dreamer):
    dyn = small_dreamer.dynamics
    B = 1
    z = torch.randn(B, 2, dyn.latent_tokens, dyn.latent_dim)
    actions = torch.randn(B, 2, dyn.action_size)
    tau_idx = torch.zeros(B, 2, dtype=torch.long)
    d_idx = torch.zeros(B, 2, dtype=torch.long)
    empty_ctx = torch.randn(B, 0, dyn.latent_tokens, dyn.latent_dim)
    empty_act = torch.randn(B, 0, dyn.action_size)
    with torch.no_grad():
        no_ctx = dyn.forward(z, tau_idx, d_idx, actions)
        zero_ctx = dyn.forward(
            z, tau_idx, d_idx, actions, context_z=empty_ctx,
            context_actions=empty_act, context_tau_idx=dyn.contextSignalIndex(),
        )
    assert torch.allclose(no_ctx, zero_ctx, atol=1e-6)


def test_context_actions_change_outputs(small_dreamer):
    """Sensitivity: changing context actions must change agent/imagine outputs,
    and a zero context is not a neutral placeholder."""
    dyn = small_dreamer.dynamics
    dyn.eval()
    B, C = 1, 3
    z = torch.randn(B, 1, dyn.latent_tokens, dyn.latent_dim)
    ctx_z = torch.randn(B, C, dyn.latent_tokens, dyn.latent_dim)
    action = torch.zeros(B, 1, dyn.action_size)
    ctx_tau = dyn.contextSignalIndex()

    ctx_a = torch.zeros(B, C, dyn.action_size)
    ctx_a[0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    ctx_b = ctx_a.clone()
    ctx_b[0, 1, 0] = 9.0
    ctx_b[0, 2, 0] = 8.0
    zeros = torch.zeros_like(ctx_a)

    with torch.no_grad():
        h_a = dyn.agentOutputs(
            z, action, context_z=ctx_z, context_tau_idx=ctx_tau, context_actions=ctx_a
        )
        h_b = dyn.agentOutputs(
            z, action, context_z=ctx_z, context_tau_idx=ctx_tau, context_actions=ctx_b
        )
        h_z = dyn.agentOutputs(
            z, action, context_z=ctx_z, context_tau_idx=ctx_tau, context_actions=zeros
        )
        assert not torch.allclose(h_a, h_b)
        assert not torch.allclose(h_a, h_z)

        torch.manual_seed(7)
        z_a = dyn.imagineLatent(ctx_z, action, K=2, context_tau_idx=ctx_tau,
                                context_actions=ctx_a)
        torch.manual_seed(7)
        z_b = dyn.imagineLatent(ctx_z, action, K=2, context_tau_idx=ctx_tau,
                                context_actions=ctx_b)
        torch.manual_seed(7)
        z_z = dyn.imagineLatent(ctx_z, action, K=2, context_tau_idx=ctx_tau,
                                context_actions=zeros)
        assert not torch.allclose(z_a, z_b)
        assert not torch.allclose(z_a, z_z)


def test_phase3_batch_one_context_one(small_dreamer):
    """Minimal B=1 / contextLength=1 / horizon=2 rollout must stay finite."""
    from attridict import AttriDict

    small_dreamer.freezePolicyPrior()
    small_dreamer.context_length = 1
    small_dreamer.imagination_horizon = 2
    batch = AttriDict(
        {
            "observations": torch.rand(1, 4, 3, 64, 64),
            "actions": torch.randn(1, 4, small_dreamer.action_size),
            "rewards": torch.randn(1, 4, 1),
            "dones": torch.zeros(1, 4, 1),
            "isRelevant": torch.ones(1, 4, 1, dtype=torch.bool),
        }
    )
    metrics = small_dreamer.trainPhase3(batch)
    assert torch.isfinite(torch.tensor(metrics["phase3Loss"]))


def test_phase3_horizon_one_is_rejected(small_dreamer, small_batch):
    """imaginationHorizon=1 has no H-1 return/advantage target; the fixed-horizon
    Phase 3 design requires H>=2.  It must fail loudly, not return a silent
    (wrong) value."""
    small_dreamer.freezePolicyPrior()
    small_dreamer.imagination_horizon = 1
    with pytest.raises((IndexError, ValueError, RuntimeError)):
        small_dreamer.trainPhase3(small_batch)


def test_phase3_frozen_params_do_not_drift(small_dreamer, small_batch):
    small_dreamer.freezePolicyPrior()
    tracked = {
        "tokenizer": small_dreamer.tokenizer,
        "dynamics": small_dreamer.dynamics,
        "rewardHead": small_dreamer.rewardHead,
        "frozenPrior": small_dreamer.frozenPrior,
    }
    before = {
        name: {k: v.detach().clone() for k, v in module.named_parameters()}
        for name, module in tracked.items()
    }
    small_dreamer.trainPhase3(small_batch)
    for name, module in tracked.items():
        for key, value in module.named_parameters():
            assert torch.equal(value.detach(), before[name][key]), (
                f"{name}.{key} drifted during Phase 3"
            )


# ---------------------------------------------------------------------------
# 3. Imagined action rollout (arriving slot)
# ---------------------------------------------------------------------------

def test_phase3_action_slot_is_arriving(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    monkeypatch.setattr(
        Independent,
        "sample",
        lambda self, sample_shape=torch.Size(): torch.zeros(
            self.batch_shape + self.event_shape, dtype=torch.long
        ),
    )
    agent_actions, imagine_actions = [], []

    def fake_agent(z, actions, context_z=None, context_tau_idx=None,
                   context_actions=None, **kwargs):
        agent_actions.append(actions.detach().clone()[:, -1:])
        B, T = z.shape[:2]
        return torch.zeros(
            B, T, small_dreamer.dynamics.model_dim, device=z.device, requires_grad=True
        )

    def fake_imagine(ctx_z, actions, K=4, context_tau_idx=None, **kwargs):
        imagine_actions.append(actions.detach().clone())
        return ctx_z[:, -1:].clone()

    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", fake_agent)
    monkeypatch.setattr(small_dreamer.dynamics, "imagineLatent", fake_imagine)

    small_dreamer.trainPhase3(small_batch)

    zero_idx = torch.zeros(1, 1, small_dreamer.action_size, dtype=torch.long)
    expected = discrete_to_actions(
        zero_idx, small_dreamer.action_low, small_dreamer.action_high, small_dreamer.action_bins
    )
    assert len(imagine_actions) == small_dreamer.imagination_horizon
    for a in imagine_actions:
        assert torch.allclose(a, expected.expand_as(a))
    # The action generated at step k becomes the current slot at step k+1.
    for k in range(1, len(agent_actions)):
        assert torch.allclose(agent_actions[k][:, 0], imagine_actions[k - 1][:, 0])


# ---------------------------------------------------------------------------
# 4. Reward / value / lambda return alignment
# ---------------------------------------------------------------------------

def test_lambda_return_alignment():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    continues = torch.full((1, 3), 0.9)
    out = computeLambdaValues(rewards, values, continues, 0.95)
    assert out.shape == (1, 3)
    bootstrap = values[0, -1]
    ref = [0.0, 0.0, 0.0]
    for i in [2, 1, 0]:
        ref[i] = rewards[0, i] + continues[0, i] * ((1 - 0.95) * values[0, i] + 0.95 * bootstrap)
        bootstrap = ref[i]
    assert torch.allclose(out[0], torch.tensor(ref), atol=1e-5)


def test_phase3_reward_value_use_current_hidden(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    reward_inputs, value_inputs = [], []

    real_reward_forward = small_dreamer.rewardHead.forward

    def spy_reward(h, distance=0):
        reward_inputs.append(h.detach().clone())
        return real_reward_forward(h, distance)

    real_value_forward = small_dreamer.valueHead.forward

    def spy_value(h):
        value_inputs.append(h.detach().clone())
        return real_value_forward(h)

    monkeypatch.setattr(small_dreamer.rewardHead, "forward", spy_reward)
    monkeypatch.setattr(small_dreamer.valueHead, "forward", spy_value)

    small_dreamer.trainPhase3(small_batch)

    B = small_batch.observations.shape[0]
    H = small_dreamer.imagination_horizon
    for h in reward_inputs:
        assert h.shape[1] == 1
    # valueHead is queried once per rollout step (B,1,D) and once for the value
    # loss over the H-1 non-bootstrap states (B,H-1,D).
    assert value_inputs[0].shape == (B, 1, small_dreamer.dynamics.model_dim)
    assert value_inputs[-1].shape == (B, H - 1, small_dreamer.dynamics.model_dim)


# ---------------------------------------------------------------------------
# 5. Gradient boundary
# ---------------------------------------------------------------------------

def test_phase3_only_policy_value_get_gradients(small_dreamer, small_batch):
    small_dreamer.freezePolicyPrior()
    small_dreamer.zero_grad(set_to_none=True)
    small_dreamer.trainPhase3(small_batch)
    for name, p in small_dreamer.tokenizer.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"tokenizer {name} got grad"
    for name, p in small_dreamer.dynamics.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"dynamics {name} got grad"
    for name, p in small_dreamer.rewardHead.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"reward {name} got grad"
    for name, p in small_dreamer.frozenPrior.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"prior {name} got grad"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in small_dreamer.policyHead.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in small_dreamer.valueHead.parameters())


# ---------------------------------------------------------------------------
# 6. Train/eval/dropout
# ---------------------------------------------------------------------------

def test_phase3_runs_frozen_modules_in_eval_and_restores(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    modes = []
    original = small_dreamer.dynamics.agentOutputs

    def spy(*args, **kwargs):
        modes.append(
            (
                small_dreamer.tokenizer.training,
                small_dreamer.dynamics.training,
                small_dreamer.frozenPrior.training,
                small_dreamer.policyHead.training,
                small_dreamer.valueHead.training,
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", spy)
    small_dreamer.tokenizer.train()
    small_dreamer.dynamics.train()
    small_dreamer.trainPhase3(small_batch)

    assert modes, "agentOutputs was not called"
    assert all(m == (False, False, False, True, True) for m in modes)
    # Restored to training after Phase 3.
    assert small_dreamer.tokenizer.training is True
    assert small_dreamer.dynamics.training is True


def test_phase3_frozen_prior_mode_restored(small_dreamer, small_batch):
    small_dreamer.freezePolicyPrior()
    small_dreamer.frozenPrior.train()
    assert small_dreamer.frozenPrior.training is True
    small_dreamer.trainPhase3(small_batch)
    assert small_dreamer.frozenPrior.training is True, (
        "trainPhase3 forced frozenPrior to eval and did not restore its mode"
    )


def test_phase3_frozen_prior_mode_restored_on_error(small_dreamer, small_batch, monkeypatch):
    small_dreamer.freezePolicyPrior()
    small_dreamer.frozenPrior.train()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(small_dreamer.dynamics, "agentOutputs", boom)
    with pytest.raises(RuntimeError):
        small_dreamer.trainPhase3(small_batch)
    assert small_dreamer.frozenPrior.training is True, (
        "frozenPrior mode not restored after an exception in trainPhase3"
    )


# ---------------------------------------------------------------------------
# 7. Formal checkpoint compatibility (read-only, opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_FORMAL_CKPT"),
    reason="set RUN_FORMAL_CKPT=1 to validate the published 320k checkpoint (read-only)",
)
def test_formal_checkpoint_readonly_compat():
    from utils import loadConfig

    assert os.path.exists(CHECKPOINT)
    before = os.stat(CHECKPOINT)
    config = loadConfig("car-racing-v4-offline.yml")
    dreamer = DreamerV4(
        (3, 64, 64),
        3,
        list(config.actionLow),
        list(config.actionHigh),
        torch.device("cpu"),
        config.dreamer,
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    # Strict per-module load: any new/removed parameter surfaces here.
    dreamer.tokenizer.load_state_dict(ckpt["tokenizer"], strict=True)
    dreamer.dynamics.load_state_dict(ckpt["dynamics"], strict=True)
    dreamer.policyHead.load_state_dict(ckpt["policyHead"], strict=True)
    dreamer.rewardHead.load_state_dict(ckpt["rewardHead"], strict=True)
    dreamer.valueHead.load_state_dict(ckpt["valueHead"], strict=True)

    dreamer.tokenizer.eval()
    dreamer.dynamics.eval()
    with torch.no_grad():
        video = torch.rand(1, 2, 3, 64, 64)
        z, _ = dreamer.tokenizer.encode(video)
        assert z.shape == (1, 2, dreamer.tokenizer.latent_tokens, dreamer.tokenizer.latent_dim)
        h = dreamer.dynamics.agentOutputs(z, torch.zeros(1, 2, 3))
        assert h.shape == (1, 2, dreamer.dynamics.model_dim)

    # Minimal Phase 3 short rollout on the real weights (small context/horizon).
    from attridict import AttriDict

    dreamer.context_length = 2
    dreamer.imagination_horizon = 2
    dreamer.freezePolicyPrior()
    dreamer.frozenPrior.load_state_dict(ckpt["frozenPrior"], strict=True)
    B, T = 1, 4
    batch = AttriDict(
        {
            "observations": torch.rand(B, T, 3, 64, 64),
            "actions": torch.randn(B, T, 3),
            "rewards": torch.randn(B, T, 1),
            "dones": torch.zeros(B, T, 1),
            "isRelevant": torch.ones(B, T, 1, dtype=torch.bool),
        }
    )
    metrics = dreamer.trainPhase3(batch)
    assert torch.isfinite(torch.tensor(metrics["phase3Loss"]))

    after = os.stat(CHECKPOINT)
    assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime), "checkpoint was modified"
