"""Causal (per-frame, temporally-causal) video tokenizer.

This is a small, paper-mechanism-driven prototype (DreamerV4-style), not a
line-by-line port of the official ``dreamer4`` reference implementation.

Terminology note: two latent representations coexist here and their element
counts are conserved (``N_b * D_b == N_z * D_z``):

* ``bottleneck_tokens`` / ``bottleneck_dim`` (config ``bottleneckTokens`` /
  ``bottleneckDim``) = ``N_b`` x ``D_b``, the tokenizer's own latent: the
  output of the tanh bottleneck read from the encoder's learned latent tokens.
* ``latent_tokens`` / ``latent_dim`` (config ``latentTokens`` / ``latentDim``)
  = ``N_z`` x ``D_z``, the representation consumed by the dynamics model: the
  bottleneck reshaped so that ``N_b`` tokens are packed into ``N_z`` spatial
  tokens of channel ``D_z``.

The reshape in ``encode``/``decode`` preserves element count and row-major
order; ``bottleneck_tokens`` is the encoder's token count, while
``latent_tokens`` is the dynamics-facing spatial token count (they differ).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from masks import _to_mask, make_block_causal_mask, make_tokenizer_encoder_mask


class CausalTokenizer(nn.Module):
    """Patch/autoencoder-style causal tokenizer for image videos.

    ``patchify -> block-causal encoder with learnable latent tokens -> tanh
    bottleneck -> block-causal decoder -> unpatchify``.

    ``encode`` and ``forward`` return the per-patch MAE mask actually used for
    this forward so the reconstruction loss can be computed only over masked
    patches (see ``loss`` / ``_reconstruction_mse``).
    """

    def __init__(self, image_size, patch_size, patch_channels, config):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_channels = patch_channels
        self.config = config
        self.num_patches = (image_size // patch_size) ** 2
        # N_b x D_b: the tokenizer's own latent (tanh bottleneck read-out).
        self.bottleneck_tokens = config.bottleneckTokens
        self.bottleneck_dim = config.bottleneckDim
        # N_z x D_z: the dynamics-facing spatial representation (N_b*D_b == N_z*D_z).
        self.latent_tokens = config.latentTokens
        self.latent_dim = config.latentDim
        self.model_dim = config.modelDim
        self.layers = config.layers
        self.heads = config.heads

        if self.bottleneck_tokens * self.bottleneck_dim != self.latent_tokens * self.latent_dim:
            raise ValueError(
                "Tokenizer bottleneck element count must be conserved: "
                f"bottleneckTokens*bottleneckDim={self.bottleneck_tokens * self.bottleneck_dim} "
                f"!= latentTokens*latentDim={self.latent_tokens * self.latent_dim}."
            )

        self.patchify = nn.Conv2d(3, patch_channels, kernel_size=patch_size, stride=patch_size)
        self.patch_proj = nn.Linear(patch_channels, self.model_dim)
        self.patch_pos = nn.Parameter(torch.randn(1, self.num_patches, self.model_dim) * 0.02)
        self.time_embed = nn.Embedding(1024, self.model_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.latent_embed = nn.Parameter(torch.randn(1, self.bottleneck_tokens, self.model_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.heads,
            dim_feedforward=self.model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.bottleneck_dim),
            nn.Tanh(),
        )
        self.latent_proj = nn.Linear(self.bottleneck_dim, self.model_dim)
        self.output_proj = nn.Linear(self.model_dim, self.patch_channels)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=self.heads,
            dim_feedforward=self.model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.layers)
        self.unpatchify = nn.ConvTranspose2d(patch_channels, 3, kernel_size=patch_size, stride=patch_size)

        self._lpips = None

    @staticmethod
    def _is_lpips_key(key):
        return key == "_lpips" or key.startswith("_lpips.") or "._lpips." in key

    def state_dict(self, *args, **kwargs):
        # ``_lpips`` is a frozen external feature extractor loaded from a fixed
        # pretrained source, so its parameters are excluded from checkpoints.
        # Remove them from the shared destination (if any) so that both direct
        # and recursive ``state_dict()`` calls stay consistent.
        state = super().state_dict(*args, **kwargs)
        for key in [key for key in state if self._is_lpips_key(key)]:
            state.pop(key)
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Incoming checkpoints never carry ``_lpips`` weights.  Drop any legacy
        # ``_lpips.*`` entries and detach a locally loaded trunk while loading so
        # ``strict=True`` keeps its meaning for every real tokenizer parameter.
        filtered = {key: value for key, value in state_dict.items() if not self._is_lpips_key(key)}
        lpips = self._modules.pop("_lpips", None)
        try:
            return super().load_state_dict(filtered, strict=strict, assign=assign)
        finally:
            if lpips is not None:
                self._modules["_lpips"] = lpips

    @torch.no_grad()
    def _load_lpips(self, device):
        if self._lpips is not None:
            return self._lpips
        from torchvision import models
        from torchvision.transforms import Normalize

        vgg = self._build_lpips_features(models, device)
        vgg.eval()
        for parameter in vgg.parameters():
            parameter.requires_grad_(False)
        self._lpips = vgg
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return self._lpips

    def _build_lpips_features(self, models, device):
        weights_path = getattr(self.config, "lpipsWeightsPath", None)
        if weights_path:
            vgg = models.vgg16(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            vgg.load_state_dict(state, strict=True)
            return vgg.features.to(device)
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception as exc:
            cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
            raise RuntimeError(
                "Failed to load the fixed pretrained VGG16 weights required by the "
                "LPIPS perceptual loss. The perceptual term must use deterministic "
                "weights and will not silently fall back to random or zero. If this "
                f"machine is offline, pre-populate the torchvision cache at '{cache_dir}' "
                "or set tokenizer.lpipsWeightsPath to a local VGG16 state_dict. To "
                "intentionally disable LPIPS, set tokenizer.lpipsWeight: 0."
            ) from exc
        return vgg.features.to(device)

    def patchify_tensor(self, video):
        B, T, C, H, W = video.shape
        flat = video.reshape(B * T, C, H, W)
        patches = self.patchify(flat)  # (B*T, patch_channels, P_h, P_w)
        patches = patches.flatten(2).transpose(1, 2)  # (B*T, P, patch_channels)
        return patches.reshape(B, T, self.num_patches, self.patch_channels)

    def encode(self, video, mask_ratio=0.0):
        B, T, C, H, W = video.shape
        patches = self.patchify_tensor(video)
        patches = self.patch_proj(patches)  # (B, T, P, D)
        time = torch.arange(T, device=video.device).unsqueeze(0)
        time_emb = self.time_embed(time).unsqueeze(2)  # (1, T, 1, D)
        patches = patches + self.patch_pos.unsqueeze(0) + time_emb

        mask = None
        if mask_ratio > 0.0:
            p = torch.rand(B, T, 1, 1, device=video.device) * mask_ratio
            mask = torch.rand(B, T, self.num_patches, 1, device=video.device) < p
            patches = torch.where(mask, self.mask_token.expand_as(patches), patches)

        latent = self.latent_embed.expand(B, T, -1, -1)  # (B, T, N_b, D)
        tokens = torch.cat((patches, latent), dim=2)  # (B, T, P+N_b, D)
        tokens = tokens.reshape(B, T * (self.num_patches + self.bottleneck_tokens), self.model_dim)

        encoder_mask = make_tokenizer_encoder_mask(
            T, self.num_patches, self.bottleneck_tokens, video.device
        )
        encoded = self.encoder(tokens, mask=encoder_mask)
        encoded = encoded.reshape(B, T, self.num_patches + self.bottleneck_tokens, self.model_dim)
        z_tokens = encoded[:, :, self.num_patches:]
        z = self.bottleneck(z_tokens)  # (B, T, N_b, D_b)
        z = z.reshape(B, T, self.latent_tokens, self.latent_dim)

        # Return the per-patch mask used by this forward as (B, T, P) bool,
        # True where masked.  ``None`` means ``mask_ratio == 0`` (no MAE).
        mask_out = None if mask is None else mask[..., 0]
        return z, mask_out

    def decode(self, z):
        B, T, N_z, D_z = z.shape
        z_tokens = z.reshape(B, T, self.bottleneck_tokens, self.bottleneck_dim)
        memory = self.latent_proj(z_tokens)  # (B, T, N_b, D)
        memory = memory.reshape(B, T * self.bottleneck_tokens, self.model_dim)

        time = torch.arange(T, device=z.device).unsqueeze(0)
        time_emb = self.time_embed(time).unsqueeze(2)  # (1, T, 1, D)
        tgt = self.patch_pos.unsqueeze(0) + time_emb
        tgt = tgt.expand(B, T, self.num_patches, self.model_dim)
        tgt = tgt.reshape(B, T * self.num_patches, self.model_dim)

        # The decoder target is the compact per-frame patch layout
        # (index i -> frame i // P, patch i % P), so its self-attention mask is
        # a plain block-causal mask over patch frames: frame t sees patch frames
        # <= t.  The previous code sliced an interleaved-layout mask, which both
        # leaked future frames and over-restricted past ones (P1-2).
        tgt_self_mask = make_block_causal_mask(T, self.num_patches, z.device)
        q_t = torch.arange(T * self.num_patches, device=z.device) // self.num_patches
        k_t = torch.arange(T * self.bottleneck_tokens, device=z.device) // self.bottleneck_tokens
        memory_mask = _to_mask(k_t[None, :] <= q_t[:, None])
        decoded = self.decoder(tgt, memory, tgt_mask=tgt_self_mask, memory_mask=memory_mask)
        decoded = decoded.reshape(B, T, self.num_patches, self.model_dim)
        # Linear is applied inside unpatchify by first returning patch channels.
        decoded = decoded.reshape(B * T, self.num_patches, self.model_dim)
        patch_features = self.output_proj(decoded)
        patch_features = patch_features.reshape(B * T, self.patch_channels, self.image_size // self.patch_size, self.image_size // self.patch_size)
        recon_flat = self.unpatchify(patch_features)  # (B*T, 3, H, W)
        return recon_flat.reshape(B, T, 3, self.image_size, self.image_size)

    def forward(self, video, mask_ratio=0.0):
        z, mask = self.encode(video, mask_ratio)
        recon = self.decode(z)
        return recon, z, mask

    def lpips(self, recon, target):
        """LPIPS VGG-feature MSE, computed frame-chunked with activation checkpointing.

        Chunking the ``B*T`` frames and recomputing each chunk's VGG16 features in
        backward keeps the peak activation footprint bounded by ``lpipsChunkFrames``
        instead of the whole batch.  The returned value stays the mean squared
        error over every feature element (``sum(chunk_sqerr) / total_elements``),
        which is numerically equivalent to ``F.mse_loss`` on the full batch.  VGG16
        features contain no stochastic layers, so the recomputed gradients are
        unchanged.
        """
        vgg = self._load_lpips(target.device)
        if vgg is None:
            return torch.tensor(0.0, device=target.device)
        B, T, C, H, W = target.shape
        a = (recon.reshape(B * T, C, H, W) * 2.0 - 1.0).clamp(-1.0, 1.0)
        b = (target.reshape(B * T, C, H, W) * 2.0 - 1.0).clamp(-1.0, 1.0)
        # VGG expects roughly [-1, 1] here; normalize with ImageNet stats.
        mean = torch.tensor([0.485, 0.456, 0.406], device=target.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=target.device).view(1, 3, 1, 1)
        a = (a - mean) / std
        b = (b - mean) / std
        trunk = vgg[:8]
        chunk = max(1, int(getattr(self.config, "lpipsChunkFrames", 64)))
        total = a.new_zeros(())
        count = 0
        for start in range(0, a.shape[0], chunk):
            a_chunk = a[start:start + chunk]
            b_chunk = b[start:start + chunk]
            if torch.is_grad_enabled() and a_chunk.requires_grad:
                fa = torch.utils.checkpoint.checkpoint(trunk, a_chunk, use_reentrant=False)
            else:
                fa = trunk(a_chunk)
            fb = trunk(b_chunk)
            total = total + (fa - fb).pow(2).sum()
            count += fa.numel()
        return total / count

    def _reconstruction_mse(self, recon, video, mask):
        """MSE between ``recon`` and ``video``.

        With ``mask`` (shape ``(B, T, P)``, True = masked) the loss is computed
        only over the masked patches, which is the MAE objective.  Each patch
        index is expanded to its ``patch_size x patch_size`` pixel region in
        the same row-major order as ``patchify_tensor`` so mask and
        reconstruction stay aligned.  With ``mask=None`` (``mask_ratio=0``) the
        fallback is full-frame MSE.
        """
        if mask is None:
            return F.mse_loss(recon, video)
        B, T, C, H, W = video.shape
        P_h = H // self.patch_size
        P_w = W // self.patch_size
        mask_img = mask.reshape(B, T, P_h, P_w)
        mask_img = mask_img.repeat_interleave(self.patch_size, dim=-2)
        mask_img = mask_img.repeat_interleave(self.patch_size, dim=-1)  # (B, T, H, W)
        diff = (recon - video) ** 2
        masked = diff * mask_img.unsqueeze(2)
        denom = mask_img.sum().float() * C
        return masked.sum() / denom.clamp_min(1.0)

    def loss(self, video, recon, z, mask=None, lossnorms=None):
        # Main reconstruction term is masked-patch MSE when a mask is provided
        # (MAE); LPIPS is intentionally kept unmasked over the full
        # reconstruction to avoid implicitly changing the perceptual term.
        mse = self._reconstruction_mse(recon, video, mask)
        if self.config.lpipsWeight > 0:
            perceptual = self.lpips(recon, video)
        else:
            perceptual = torch.zeros((), device=video.device, dtype=video.dtype)
        metrics = {"mse": mse.item(), "lpips": perceptual.item()}
        if lossnorms is not None:
            mse_n = lossnorms["mse"](mse)
            lpips_n = lossnorms["lpips"](perceptual)
            total = mse_n + self.config.lpipsWeight * lpips_n
        else:
            total = mse + self.config.lpipsWeight * perceptual
        metrics["tokenizerLoss"] = total.item()
        return total, metrics
