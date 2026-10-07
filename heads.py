import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, kl_divergence

from twohot import TwoHotDist


class PolicyHead(nn.Module):
    """Discretized continuous policy: action_size x action_bins categories."""

    def __init__(self, input_size, action_size, action_bins, mtp_length, hidden_size):
        super().__init__()
        self.action_size = action_size
        self.action_bins = action_bins
        self.mtp_length = mtp_length
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, action_size * action_bins) for _ in range(mtp_length + 1)]
        )

    def _dist(self, logits):
        logits = logits.reshape(*logits.shape[:-1], self.action_size, self.action_bins)
        return Independent(Categorical(logits=logits), 1)

    def forward(self, h, distance=0):
        return self._dist(self.heads[distance](self.network(h)))

    def forwardAll(self, h):
        shared = self.network(h)
        return [self._dist(head(shared)) for head in self.heads]


class RewardHead(nn.Module):
    """MTP reward head producing symexp two-hot distributions."""

    def __init__(self, input_size, mtp_length, hidden_size, bins, low, high):
        super().__init__()
        self.mtp_length = mtp_length
        self.bins = bins
        self.low = low
        self.high = high
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, bins) for _ in range(mtp_length + 1)])

    def forward(self, h, distance=0):
        return TwoHotDist(self.heads[distance](self.network(h)), self.bins, self.low, self.high)

    def forwardAll(self, h):
        shared = self.network(h)
        return [TwoHotDist(head(shared), self.bins, self.low, self.high) for head in self.heads]


class ValueHead(nn.Module):
    def __init__(self, input_size, hidden_size, bins, low, high):
        super().__init__()
        self.bins = bins
        self.low = low
        self.high = high
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, bins),
        )

    def forward(self, h):
        return TwoHotDist(self.network(h), self.bins, self.low, self.high)


def discretize_actions(actions, action_low, action_high, action_bins):
    low = torch.as_tensor(action_low, device=actions.device, dtype=actions.dtype)
    high = torch.as_tensor(action_high, device=actions.device, dtype=actions.dtype)
    norm = ((actions - low) / (high - low + 1e-6)).clamp(0.0, 1.0)
    return (norm * (action_bins - 1)).round().long()


def discrete_to_actions(indices, action_low, action_high, action_bins):
    low = torch.as_tensor(action_low, device=indices.device, dtype=torch.float32)
    high = torch.as_tensor(action_high, device=indices.device, dtype=torch.float32)
    norm = indices.float() / (action_bins - 1)
    return norm * (high - low) + low


def pmpo_loss(logp, policy_dist, prior_dist, advantages, alpha=0.5, beta=0.3):
    """Prompt equation (11): PMPO with reverse KL to a frozen prior."""
    logp = logp.flatten()
    advantages = advantages.flatten()
    pos = advantages >= 0.0
    neg = ~pos
    zero = torch.tensor(0.0, device=logp.device, dtype=logp.dtype)
    neg_term = logp[neg].mean() if neg.any() else zero
    pos_term = logp[pos].mean() if pos.any() else zero
    policy_term = (1.0 - alpha) * neg_term - alpha * pos_term
    kl = kl_divergence(policy_dist, prior_dist)
    return policy_term + beta * kl.mean()
