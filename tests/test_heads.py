import copy

import torch
from torch.distributions import Categorical, Independent, kl_divergence

import heads
from heads import PolicyHead, RewardHead, ValueHead, discrete_to_actions, discretize_actions, pmpo_loss


def make_categorical_dist(n, action_size=3, bins=5):
    logits = torch.randn(n, action_size, bins)
    return Independent(Categorical(logits=logits), 1)


def test_policyhead_output_dist():
    head = PolicyHead(input_size=64, action_size=3, action_bins=5, mtp_length=2, hidden_size=64)
    h = torch.randn(2, 4, 64)
    dist = head(h)
    assert isinstance(dist, Independent)
    assert dist.event_shape == torch.Size([3])


def test_policyhead_mtp_heads():
    small = PolicyHead(input_size=64, action_size=3, action_bins=5, mtp_length=2, hidden_size=64)
    assert len(small.forwardAll(torch.randn(2, 4, 64))) == 3

    nine = PolicyHead(input_size=64, action_size=3, action_bins=5, mtp_length=8, hidden_size=64)
    assert len(nine.forwardAll(torch.randn(2, 4, 64))) == 9


def test_rewardhead_twohot():
    head = RewardHead(input_size=64, mtp_length=2, hidden_size=64, bins=64, low=-5.0, high=5.0)
    dist = head(torch.randn(2, 4, 64), distance=0)
    assert dist.bins == 64
    assert dist.logits.shape[-1] == 64


def test_valuehead_twohot():
    head = ValueHead(input_size=64, hidden_size=64, bins=64, low=-5.0, high=5.0)
    dist = head(torch.randn(2, 4, 64))
    assert dist.bins == 64
    assert dist.logits.shape[-1] == 64


def test_discretize_actions_roundtrip():
    torch.manual_seed(0)
    actions = torch.rand(100, 3) * 1.8 - 0.9
    action_low = [-1.0, -1.0, -1.0]
    action_high = [1.0, 1.0, 1.0]
    indices = discretize_actions(actions, action_low, action_high, 5)
    restored = discrete_to_actions(indices, action_low, action_high, 5)
    assert (restored - actions).abs().max().item() < (1.0 - (-1.0)) / 5


def test_pmpo_all_positive():
    B, T = 4, 8
    logp = torch.randn(B * T, requires_grad=True)
    policy_dist = make_categorical_dist(B * T)
    prior_dist = make_categorical_dist(B * T)
    loss = pmpo_loss(logp, policy_dist, prior_dist, torch.ones(B * T))
    assert loss.dim() == 0
    loss.backward()


def test_pmpo_all_negative():
    B, T = 4, 8
    logp = torch.randn(B * T, requires_grad=True)
    policy_dist = make_categorical_dist(B * T)
    prior_dist = make_categorical_dist(B * T)
    loss = pmpo_loss(logp, policy_dist, prior_dist, -torch.ones(B * T))
    assert loss.dim() == 0
    loss.backward()


def test_pmpo_mixed():
    B, T = 4, 8
    advantages = torch.cat((torch.ones(B * T // 2), -torch.ones(B * T - B * T // 2)))
    logp = torch.randn(B * T, requires_grad=True)
    policy_dist = make_categorical_dist(B * T)
    prior_dist = make_categorical_dist(B * T)
    loss = pmpo_loss(logp, policy_dist, prior_dist, advantages)
    assert loss.dim() == 0
    loss.backward()


def test_pmpo_returns_scalar():
    n = 32
    logp = torch.randn(n, requires_grad=True)
    policy_dist = make_categorical_dist(n)
    prior_dist = make_categorical_dist(n)
    advantages = torch.randn(n)
    loss = pmpo_loss(logp, policy_dist, prior_dist, advantages)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_pmpo_kl_is_reverse(monkeypatch):
    n = 16
    logp = torch.randn(n)
    policy_dist = make_categorical_dist(n)
    prior_dist = make_categorical_dist(n)
    calls = []

    def fake_kl(p, q):
        calls.append((p, q))
        return kl_divergence(p, q)

    monkeypatch.setattr(heads, "kl_divergence", fake_kl)
    pmpo_loss(logp, policy_dist, prior_dist, torch.ones(n))
    assert calls
    assert calls[0][0] is policy_dist
    assert calls[0][1] is prior_dist


def test_pmpo_empty_positive_set_term():
    n = 8
    logp = torch.arange(n, dtype=torch.float32)
    policy_dist = make_categorical_dist(n)
    prior_dist = make_categorical_dist(n)
    advantages = -torch.ones(n)  # all negative -> positive set empty
    alpha, beta = 0.5, 0.3
    loss = pmpo_loss(logp, policy_dist, prior_dist, advantages, alpha=alpha, beta=beta)
    expected_policy = (1.0 - alpha) * logp.mean()
    kl = kl_divergence(policy_dist, prior_dist).mean()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected_policy + beta * kl)


def test_pmpo_backward():
    head = PolicyHead(input_size=64, action_size=3, action_bins=5, mtp_length=2, hidden_size=64)
    h = torch.randn(2, 4, 64)
    policy_dist = head(h)
    prior_head = copy.deepcopy(head)
    prior_head.requires_grad_(False)
    prior_dist = prior_head(h.detach())
    indices = torch.zeros(2, 4, 3, dtype=torch.long)
    logp = policy_dist.log_prob(indices)
    advantages = torch.ones(2, 4)
    loss = pmpo_loss(logp, policy_dist, prior_dist, advantages)
    loss.backward()
    assert head.network[0].weight.grad is not None
