import torch
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotDist:
    """Two-hot encoded distribution for scalar prediction.

    Values are first transformed with symlog, then represented by two
    adjacent bins.  `log_prob`, `mean`, and `sample` are provided for
    training/planning.
    """

    def __init__(self, logits, bins, low, high):
        self.logits = logits
        self.bins = bins
        self.low = low
        self.high = high
        self.cut_points = torch.linspace(
            low, high, bins, device=logits.device, dtype=logits.dtype
        )
        self.bin_width = (high - low) / (bins - 1)

    @property
    def probs(self):
        return self.logits.softmax(-1)

    @property
    def mean_symlog(self):
        return (self.probs * self.cut_points).sum(-1)

    @property
    def mean(self):
        return symexp(self.mean_symlog)

    @property
    def mode(self):
        idx = self.logits.argmax(-1)
        return symexp(self.cut_points[idx])

    def log_prob(self, value):
        value_sym = symlog(value).clamp(self.low, self.high)
        below = ((value_sym - self.low) / self.bin_width).long().clamp(0, self.bins - 2)
        above = below + 1
        prop = (value_sym - self.cut_points[below]) / self.bin_width

        log_probs = F.log_softmax(self.logits, dim=-1)
        lp = (1.0 - prop) * log_probs.gather(-1, below.unsqueeze(-1)).squeeze(-1)
        lp = lp + prop * log_probs.gather(-1, above.unsqueeze(-1)).squeeze(-1)
        return lp

    def sample(self):
        probs = self.probs
        idx = torch.multinomial(probs.reshape(-1, self.bins), 1, replacement=True)
        idx = idx.reshape(probs.shape[:-1])
        return symexp(self.cut_points[idx])


def twohot_encode(value, bins, low, high):
    value_sym = symlog(value).clamp(low, high)
    bin_width = (high - low) / (bins - 1)
    cut_points = torch.linspace(low, high, bins, device=value.device, dtype=value.dtype)
    below = ((value_sym - low) / bin_width).long().clamp(0, bins - 2)
    above = below + 1
    prop = (value_sym - cut_points[below]) / bin_width

    target = torch.zeros(*value_sym.shape, bins, device=value.device, dtype=value.dtype)
    target.scatter_(-1, below.unsqueeze(-1), (1.0 - prop).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), prop.unsqueeze(-1))
    return target
