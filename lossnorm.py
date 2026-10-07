import torch
import torch.nn as nn


class LossNormalizer(nn.Module):
    """EMA-based RMS normalization for heterogeneous loss terms."""

    def __init__(self, device, decay=0.99, eps=1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("rms", torch.ones((), dtype=torch.float32, device=device))

    def forward(self, loss):
        loss_detached = loss.detach()
        current_rms = torch.sqrt((loss_detached ** 2).mean() + self.eps)
        self.rms = self.decay * self.rms + (1.0 - self.decay) * current_rms
        return loss / (self.rms.detach() + self.eps)
