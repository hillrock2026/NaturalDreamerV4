import sys
from pathlib import Path

import pytest
import torch
from attridict import AttriDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def fast_lpips(request, monkeypatch):
    """Avoid loading a full VGG LPIPS network in the fast unit tests.

    Tests that need the real LPIPS path declare the ``real_lpips`` fixture,
    which opts out of this patch (detected via ``request.fixturenames`` so the
    opt-out works even though this fixture is autouse).  This keeps the fast
    suite fast while still allowing isolated real-LPIPS coverage.
    """
    if "real_lpips" in request.fixturenames:
        return
    from tokenizer import CausalTokenizer

    def zero_lpips(self, recon, target):
        return torch.tensor(0.0, device=target.device, dtype=target.dtype)

    monkeypatch.setattr(CausalTokenizer, "lpips", zero_lpips)


@pytest.fixture
def real_lpips():
    """Opt out of the autouse ``fast_lpips`` zero-LPIPS patch."""
    return None


@pytest.fixture
def small_config():
    return AttriDict(
        {
            "imageSize": 64,
            "patchSize": 8,
            "patchChannels": 32,
            "contextLength": 4,
            "imaginationHorizon": 3,
            "mtpLength": 2,
            "actionBins": 5,
            "tokenizer": AttriDict(
                {
                    "bottleneckTokens": 128,
                    "bottleneckDim": 16,
                    "latentTokens": 64,
                    "latentDim": 32,
                    "modelDim": 64,
                    "layers": 2,
                    "heads": 2,
                    "mlpRatio": 2,
                    "maskRatioMax": 0.9,
                    "lpipsWeight": 0.2,
                    "singleFrameProb": 0.3,
                }
            ),
            "dynamics": AttriDict(
                {
                    "modelDim": 64,
                    "layers": 4,
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
            "discount": 0.997,
            "lambda_": 0.95,
            "pmpoAlpha": 0.5,
            "pmpoBeta": 0.3,
            "twohot": AttriDict({"bins": 64, "low": -5.0, "high": 5.0}),
            "headHidden": 64,
            "lr": 1e-4,
            "phase2Lr": 3e-5,
            "weightDecay": 0.01,
            "gradientClip": 1.0,
            "gradientNormType": 2,
            "precision": "fp32",
            "lossNormDecay": 0.99,
            "buffer": AttriDict({"capacity": 1000}),
        }
    )


@pytest.fixture
def small_dreamer(small_config):
    from dreamer import DreamerV4

    return DreamerV4(
        (3, 64, 64),
        3,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        torch.device("cpu"),
        small_config,
    )


@pytest.fixture
def small_batch():
    B, T = 2, 8
    is_relevant = torch.zeros(B, T, 1, dtype=torch.bool)
    is_relevant[:, : T // 2] = True  # mixed: first half relevant, second half uniform
    return AttriDict(
        {
            "observations": torch.rand(B, T, 3, 64, 64),
            "actions": torch.randn(B, T, 3),
            "rewards": torch.randn(B, T, 1),
            "dones": torch.zeros(B, T, 1),
            "isRelevant": is_relevant,
        }
    )
