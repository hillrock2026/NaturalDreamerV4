import torch

from lossnorm import LossNormalizer


def test_rms_ema_updates():
    norm = LossNormalizer(torch.device("cpu"), decay=0.99)
    loss = torch.tensor(3.0)
    for _ in range(1000):
        norm(loss)
    assert torch.allclose(norm.rms, torch.tensor(3.0), atol=1e-2)


def test_normalizes_scale():
    norm_large = LossNormalizer(torch.device("cpu"), decay=0.99)
    norm_small = LossNormalizer(torch.device("cpu"), decay=0.99)
    for _ in range(1000):
        out_large = norm_large(torch.tensor(100.0))
        out_small = norm_small(torch.tensor(0.01))
    assert torch.allclose(out_large.abs(), torch.ones(()), atol=0.05)
    assert torch.allclose(out_small.abs(), torch.ones(()), atol=0.05)


def test_detach_blocks_grad():
    norm = LossNormalizer(torch.device("cpu"), decay=0.99)
    loss = torch.tensor(2.0, requires_grad=True)
    out = norm(loss)
    out.backward()
    assert loss.grad is not None
    assert norm.rms.grad is None


def test_multi_key_independent():
    first = LossNormalizer(torch.device("cpu"), decay=0.99)
    second = LossNormalizer(torch.device("cpu"), decay=0.99)
    first(torch.tensor(5.0))
    second(torch.tensor(-0.5))
    assert not torch.allclose(first.rms, second.rms)
