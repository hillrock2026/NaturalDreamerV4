import torch
import torch.nn.functional as F

from twohot import TwoHotDist, symexp, symlog, twohot_encode


def test_symlog_symexp_roundtrip():
    for val in [-100.0, -1.0, 0.0, 0.5, 50.0]:
        x = torch.tensor(val)
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)


def test_twohot_log_prob_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(3, 4, 64, requires_grad=True)
    dist = TwoHotDist(logits, bins=64, low=-5.0, high=5.0)
    value = torch.tensor([[-4.5, -2.0, 0.0, 3.2], [-3.0, 1.0, 2.0, 4.0], [-1.0, 0.1, 0.5, 3.5]])

    target = twohot_encode(value, bins=64, low=-5.0, high=5.0)
    log_probs = F.log_softmax(logits, dim=-1)
    manual = (target * log_probs).sum(dim=-1)
    assert torch.allclose(dist.log_prob(value), manual, atol=1e-6)


def test_twohot_mean_is_symexp_of_weighted_sum():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 64)
    dist = TwoHotDist(logits, bins=64, low=-5.0, high=5.0)
    expected = symexp((dist.probs * dist.cut_points).sum(dim=-1))
    assert torch.allclose(dist.mean, expected, atol=1e-6)


def test_twohot_sample_in_range():
    torch.manual_seed(0)
    logits = torch.randn(1000, 64)
    dist = TwoHotDist(logits, bins=64, low=-5.0, high=5.0)
    samples = dist.sample()
    assert samples.min().item() >= symexp(torch.tensor(-5.0)).item() - 1e-6
    assert samples.max().item() <= symexp(torch.tensor(5.0)).item() + 1e-6


def test_twohot_endpoints_and_clamping():
    torch.manual_seed(0)
    logits = torch.randn(1, 64, requires_grad=True)
    dist = TwoHotDist(logits, bins=64, low=-5.0, high=5.0)
    log_probs = F.log_softmax(logits, dim=-1)
    # low/high are symlog-space bounds; feed real values whose symlog lands on
    # the endpoints.
    low_value = symexp(torch.tensor(-5.0))
    high_value = symexp(torch.tensor(5.0))
    lp_low = dist.log_prob(low_value.reshape(1))
    assert torch.allclose(lp_low, log_probs[0, 0])
    # Far below low is clamped to the low endpoint bin.
    assert torch.allclose(dist.log_prob(torch.tensor([-1e6])), lp_low)
    lp_high = dist.log_prob(high_value.reshape(1))
    assert torch.allclose(lp_high, log_probs[0, -1])
    # Far above high is clamped to the high endpoint bin.
    assert torch.allclose(dist.log_prob(torch.tensor([1e6])), lp_high)


def test_twohot_mean_negative_values():
    torch.manual_seed(0)
    logits = torch.randn(4, 64)
    dist = TwoHotDist(logits, bins=64, low=-5.0, high=5.0)
    mean = dist.mean
    assert torch.isfinite(mean).all()
    # symexp domain means the mean can be negative but bounded by the endpoints.
    assert (mean >= symexp(torch.tensor(-5.0)) - 1e-6).all()
    assert (mean <= symexp(torch.tensor(5.0)) + 1e-6).all()


def test_twohot_encode_is_two_hot():
    value = torch.tensor([-3.0, -1.0, 0.5, 2.5])
    target = twohot_encode(value, bins=64, low=-5.0, high=5.0)
    assert torch.allclose(target.sum(dim=-1), torch.ones_like(target.sum(dim=-1)))
    assert (target > 0).sum(dim=-1).tolist() == [2, 2, 2, 2]


def test_twohot_log_prob_backward():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 64, requires_grad=True)
    dist = TwoHotDist(logits, bins=64, low=-5.0, high=5.0)
    value = torch.tensor([[-4.0, 0.0, 1.0], [-2.0, 2.0, 3.0]])
    dist.log_prob(value).sum().backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
