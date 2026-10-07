import torch
import torch.nn as nn

from masks import make_dynamics_mask
from transformer import TransformerBackbone


def build_legal_tau_d_pairs(tau_values, step_bins):
    """Enumerate every (tau_idx, d_idx) pair that satisfies ``tau + d <= 1``.

    The shortcut-forcing sampling grid must only contain jointly legal
    (signal level, step size) combinations: requesting a step ``d`` from a
    signal level ``tau`` may never cross the clean state ``tau = 1``.
    """
    pairs = []
    for d_idx, d in enumerate(step_bins):
        for tau_idx, tau in enumerate(tau_values):
            if tau + d <= 1.0 + 1e-9:
                pairs.append((tau_idx, d_idx))
    if not pairs:
        raise ValueError(
            "No legal (tau, d) pairs satisfy tau + d <= 1 for "
            f"tau_values={list(tau_values)} and step_bins={list(step_bins)}."
        )
    return pairs


def sample_tau_d_from_pairs(legal_pairs, shape, device):
    """Uniformly sample ``(tau_idx, d_idx)`` from ``legal_pairs``.

    Returns two long tensors of shape ``shape``.
    """
    tau_table = torch.tensor([p[0] for p in legal_pairs], dtype=torch.long)
    d_table = torch.tensor([p[1] for p in legal_pairs], dtype=torch.long)
    flat = torch.randint(0, len(legal_pairs), (shape[0] * shape[1],), device=device)
    tau_idx = tau_table.to(device)[flat].reshape(*shape)
    d_idx = d_table.to(device)[flat].reshape(*shape)
    return tau_idx, d_idx


def validate_step_bins(step_bins):
    """Validate that ``step_bins`` is a recursively-bisecting sequence.

    The shortcut-forcing teacher uses two half-steps of size ``d / 2``, so
    every non-minimum step must have its half present in the grid.  We only
    support power-of-two bisection (each entry is twice the previous), which
    makes the half-step equal to the previous bin exactly and avoids silently
    approximating ``d / 2`` with a nearest neighbour.
    """
    if not step_bins:
        raise ValueError("step_bins must be non-empty.")
    if len(step_bins) == 1:
        if step_bins[0] <= 0:
            raise ValueError(f"step_bins must be positive, got {list(step_bins)}.")
        return list(step_bins)
    for i in range(1, len(step_bins)):
        prev = step_bins[i - 1]
        curr = step_bins[i]
        if prev <= 0 or curr <= prev:
            raise ValueError(
                f"step_bins must be strictly increasing and positive, got {list(step_bins)}."
            )
        if abs(curr - 2.0 * prev) > 1e-9 * max(1.0, abs(curr)):
            raise ValueError(
                f"step_bins must be recursively bisecting (each step is twice the "
                f"previous) so that d/2 is representable, but {curr} != 2 * {prev} "
                f"in {list(step_bins)}."
            )
    return list(step_bins)


class DynamicsTransformer(nn.Module):
    """Interactive latent dynamics with shortcut/flow-matching forcing."""

    def __init__(self, latent_tokens, latent_dim, action_size, config):
        super().__init__()
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.action_size = action_size
        self.config = config
        # Context corruption level (paper's ``tau_ctx``).  The dreamer-level
        # config carries it; fall back to the design default.
        self.tau_ctx = float(getattr(config, "tauCtx", 0.1))
        self.model_dim = config.modelDim
        self.registers = config.registers
        self.action_tokens = config.actionTokens
        self.agent_tokens_count = config.agentTokens
        self.tau_bins = config.tauBins
        self.step_bins = validate_step_bins(list(config.stepBins))
        self.tau_values = [i / (self.tau_bins - 1) for i in range(self.tau_bins)]
        # Kept as a validation aid (raises if the grid has no jointly-legal
        # pairs); the actual sampling uses the conditional grid in
        # ``_sample_tau_d``.
        self._legal_pairs = build_legal_tau_d_pairs(self.tau_values, self.step_bins)

        self.z_proj = nn.Linear(latent_dim, self.model_dim)
        self.z_pos = nn.Parameter(torch.randn(1, latent_tokens, self.model_dim) * 0.02)
        self.action_proj = nn.Linear(action_size, self.model_dim)
        self.action_embed = nn.Parameter(torch.randn(1, self.action_tokens, self.model_dim) * 0.02)
        self.register_embed = nn.Parameter(torch.randn(1, self.registers, self.model_dim) * 0.02)
        self.tau_embed = nn.Embedding(self.tau_bins, self.model_dim)
        self.d_embed = nn.Embedding(len(self.step_bins), self.model_dim)
        self.agent_embed = nn.Parameter(torch.randn(1, self.agent_tokens_count, self.model_dim) * 0.02)

        # ``gradientCheckpointing`` defaults to True: the formal long-sequence
        # (T=64) Phase 2 pass would otherwise retain one (B,H,L,L) attention
        # tensor per block and OOM on a 48 GB GPU.  The flag only trades compute
        # for memory; it does not change the forward math.
        self.gradient_checkpointing = bool(getattr(config, "gradientCheckpointing", True))
        self.backbone = TransformerBackbone(
            dim=self.model_dim,
            layers=config.layers,
            heads=config.heads,
            kv_heads=config.kvHeads,
            mlp_ratio=4,
            time_every=4,
            soft_cap=config.softCap,
            dropout=config.dropout,
            max_T=512,
            max_S=512,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.z_out = nn.Linear(self.model_dim, latent_dim)
        self.agent_out = nn.Linear(self.model_dim, self.model_dim)

    def _tau_index(self, tau):
        vals = torch.tensor(self.tau_values, device=tau.device, dtype=tau.dtype)
        return (tau.unsqueeze(-1) - vals).abs().argmin(-1)

    def _tau_value(self, tau_idx):
        vals = torch.tensor(self.tau_values, device=tau_idx.device, dtype=torch.float32)
        return vals[tau_idx]

    def _step_value(self, d_idx):
        vals = torch.tensor(self.step_bins, device=d_idx.device, dtype=torch.float32)
        return vals[d_idx]

    def _sample_tau_d(self, B, T, device):
        """Sample ``(tau_idx, d_idx)`` on the step-aligned conditional grid.

        First a step ``d`` is sampled uniformly over ``step_bins``; then
        ``tau`` is sampled uniformly from the grid aligned to that step,
        ``tau in {0, d, 2d, ..., 1 - d}``, mirroring the paper/reference
        shortcut sampling (for ``d = 1/K`` the signal level is drawn from the
        ``K``-point grid).  This guarantees ``tau + d <= 1`` and
        ``tau + d/2 <= 1`` by construction, without independently sampling
        ``tau`` and ``d``.
        """
        d_idx = torch.randint(0, len(self.step_bins), (B, T), device=device)
        d = self._step_value(d_idx)  # (B, T)
        K = torch.round(1.0 / d).clamp_min(1).long()  # 1/d, exact for bisecting bins
        j = (torch.rand(B, T, device=device) * K.float()).floor().long()  # U{0..K-1}
        tau = j.float() * d
        tau_idx = self._tau_index(tau)
        return tau_idx, d_idx

    def _frame_tokens(self, z, tau_idx, d_idx, actions, agent_tokens=None):
        B, T, N, D = z.shape
        z = self.z_proj(z) + self.z_pos.unsqueeze(0)
        a = self.action_proj(actions).unsqueeze(2) + self.action_embed.unsqueeze(0)
        reg = self.register_embed.expand(B, T, -1, -1)
        tau = self.tau_embed(tau_idx).unsqueeze(2)
        d = self.d_embed(d_idx).unsqueeze(2)
        tokens = torch.cat((a, reg, tau, d, z), dim=2)
        if agent_tokens is not None:
            tokens = torch.cat((tokens, agent_tokens), dim=2)
        return tokens

    def contextSignalIndex(self, tau_ctx=None):
        """Return the context signal index used for the corrupted history.

        The paper corrupts the history to signal level ``1 - tau_ctx``.  The
        project discretises tau on ``tauBins`` levels and only trains pairs with
        ``tau + d <= 1``; for the minimum step the largest *trained* signal is
        ``1 - stepBins[0]`` (0.75 on the default grid).  The requested signal is
        therefore snapped into that legal/trained range so the context frame
        uses a tau embedding the model actually saw during shortcut training.
        ``tau_ctx <= 0`` disables corruption and tags the context as clean.
        """
        if tau_ctx is None:
            tau_ctx = self.tau_ctx
        if tau_ctx <= 0.0:
            return self.tau_bins - 1
        signal = 1.0 - float(tau_ctx)
        max_signal = 1.0 - self.step_bins[0]
        signal = min(signal, max_signal)
        idx = int(round(signal * (self.tau_bins - 1)))
        return max(0, min(idx, self.tau_bins - 1))

    def prepareContext(self, z, tau_ctx=None):
        """Corrupt ``z`` to the context signal level and return ``(ctx, tau_idx)``.

        Both ``agentOutputs`` and ``imagineLatent`` in Phase 3 consume the
        *same* returned tensor so the hidden state and the generated next frame
        are conditioned on one shared history distribution.
        """
        tau_idx = self.contextSignalIndex(tau_ctx)
        signal = self.tau_values[tau_idx]
        if signal >= 1.0:
            return z, tau_idx
        ctx = signal * z + (1.0 - signal) * torch.randn_like(z)
        return ctx, tau_idx

    def _forward_tokens(self, z, tau_idx, d_idx, actions, agent_tokens=None,
                        context_z=None, context_tau_idx=None, context_actions=None):
        current = self._frame_tokens(z, tau_idx, d_idx, actions, agent_tokens)
        if context_z is not None:
            B, C, N, D = context_z.shape
            if context_tau_idx is None:
                context_tau_idx = self.contextSignalIndex()
            # A non-empty context always carries the real arriving actions of
            # its frames.  ``context_z[t] <-> context_actions[t]``; silently
            # substituting zeros would condition the model on a fake action
            # history (the P1-A bug), so it is a hard error here.
            if context_actions is None:
                raise ValueError(
                    "context_actions must be provided whenever context_z is "
                    "not None: context frame t must carry its arriving action "
                    "(refusing to silently substitute all-zero actions)."
                )
            if context_actions.shape[0] != B or context_actions.shape[1] != C:
                raise ValueError(
                    "context_actions must align with context_z on batch and time "
                    f"dims: got context_z={tuple(context_z.shape)} but "
                    f"context_actions={tuple(context_actions.shape)}."
                )
            if context_actions.shape[-1] != self.action_size:
                raise ValueError(
                    f"context_actions last dim must be action_size="
                    f"{self.action_size}, got {tuple(context_actions.shape)}."
                )
            # Device/dtype must match the current-frame tensor (``z``); a
            # mismatch would otherwise fail deep inside ``action_proj`` /
            # ``torch.cat`` with an opaque RuntimeError.  Fail fast, never cast
            # or move silently.
            if context_actions.device != z.device:
                raise ValueError(
                    f"context_actions device must match z device: expected "
                    f"{z.device}, got {context_actions.device}."
                )
            if context_actions.dtype != z.dtype:
                raise ValueError(
                    f"context_actions dtype must match z dtype: expected "
                    f"{z.dtype}, got {context_actions.dtype}."
                )
            ctx_tau = torch.full((B, C), int(context_tau_idx), device=z.device, dtype=torch.long)
            ctx_d = torch.zeros((B, C), dtype=torch.long, device=z.device)
            ctx_actions = context_actions
            ctx_agent = None
            if agent_tokens is not None:
                ctx_agent = self.agent_embed.expand(B, C, -1, -1)
            context = self._frame_tokens(context_z, ctx_tau, ctx_d, ctx_actions, ctx_agent)
            tokens = torch.cat((context, current), dim=1)
        else:
            tokens = current

        B, T_total, S_total, D = tokens.shape
        x = tokens.reshape(B, T_total * S_total, D)
        x = self.backbone(x, T_total, S_total)
        x = x.reshape(B, T_total, S_total, D)
        return x, T_total, S_total

    def forward(self, z_tilde, tau_idx, d_idx, actions, agent_tokens=None,
                context_z=None, context_tau_idx=None, context_actions=None):
        current_T = z_tilde.shape[1]
        x, T_total, S_total = self._forward_tokens(
            z_tilde, tau_idx, d_idx, actions, agent_tokens, context_z,
            context_tau_idx, context_actions,
        )
        non_agent = self.action_tokens + self.registers + 2
        z_out = x[:, :, non_agent:non_agent + self.latent_tokens]
        return self.z_out(z_out)[:, -current_T:]

    def agentOutputs(self, z, actions, context_z=None, context_tau_idx=None,
                     context_actions=None):
        current_T = z.shape[1]
        B, T, N, D = z.shape
        tau_idx = torch.full((B, T), self.tau_bins - 1, device=z.device)
        d_idx = torch.zeros((B, T), dtype=torch.long, device=z.device)
        agent_tokens = self.agent_embed.expand(B, T, -1, -1)
        x, _, S_total = self._forward_tokens(
            z, tau_idx, d_idx, actions, agent_tokens, context_z,
            context_tau_idx, context_actions,
        )
        agent = x[:, :, S_total - self.agent_tokens_count:]
        agent = agent.mean(dim=2)
        return self.agent_out(agent)[:, -current_T:]

    def shortcutForcingLoss(self, z1, actions):
        """Shortcut-forcing objective (Eqs. 6-8), returning per-sample losses.

        Sampling draws ``(tau, d)`` jointly on the legal grid so every sample
        satisfies ``tau + d <= 1`` and the half-step teacher satisfies
        ``tau + d / 2 <= 1``.  Samples at the minimum step use the true
        regression target ``||z_hat1 - z1||^2``; larger steps use the
        stop-gradient two-half-step bootstrap target.  Returns ``(per_sample,
        metrics)`` where ``per_sample`` has shape ``(B, T)`` so callers can
        apply relevance masks before reducing.
        """
        B, T, N, D = z1.shape
        tau_idx, d_idx = self._sample_tau_d(B, T, z1.device)

        tau = self._tau_value(tau_idx)  # (B, T)
        d = self._step_value(d_idx)     # (B, T)

        if not bool((tau + d <= 1.0 + 1e-6).all()):
            raise RuntimeError(
                "Sampled (tau, d) violates tau + d <= 1: "
                f"tau={tau}, d={d}."
            )

        z0 = torch.randn_like(z1)
        z_tilde = (1.0 - tau[..., None, None]) * z0 + tau[..., None, None] * z1

        pred = self.forward(z_tilde, tau_idx, d_idx, actions)

        d_is_min = (d_idx == 0)

        denom = (1.0 - tau[..., None, None]).clamp_min(1e-6)
        flow_pred = (pred - z_tilde) / denom

        regression = ((pred - z1) ** 2).mean(dim=(-1, -2))

        # Teacher: two half-steps, stop-gradient (Eq. 7).  step_bins is
        # validated to be recursively bisecting, so d / 2 is exactly the
        # previous bin and half_idx = d_idx - 1 (no nearest-neighbour).
        with torch.no_grad():
            half_idx = (d_idx - 1).clamp_min(0)
            half_d = self._step_value(half_idx)
            tau_mid = tau + half_d
            if not bool((tau_mid <= 1.0 + 1e-6).all()):
                raise RuntimeError(
                    "Half-step teacher violates tau + d / 2 <= 1: "
                    f"tau_mid={tau_mid}, half_d={half_d}."
                )
            tau_mid_idx = self._tau_index(tau_mid)

            first = self.forward(z_tilde, tau_idx, half_idx, actions)
            b1 = (first - z_tilde) / denom
            z_prime = z_tilde + b1 * half_d[..., None, None]
            second = self.forward(z_prime, tau_mid_idx, half_idx, actions)
            denom_mid = (1.0 - tau_mid[..., None, None]).clamp_min(1e-6)
            b2 = (second - z_prime) / denom_mid
            teacher = (b1 + b2) / 2.0

        bootstrap = ((1.0 - tau) ** 2) * ((flow_pred - teacher) ** 2).mean(dim=(-1, -2))

        w = 0.9 * tau + 0.1
        per_sample = torch.where(d_is_min, w * regression, w * bootstrap)

        not_min = ~d_is_min
        bootstrap_metric = bootstrap[not_min].mean().item() if not_min.any() else 0.0

        metrics = {
            "flowLoss": per_sample.mean().item(),
            "flowRegression": regression.mean().item(),
            "flowBootstrap": bootstrap_metric,
        }
        return per_sample, metrics

    def imagineLatent(self, ctx_z, actions, K=4, context_tau_idx=None,
                      context_actions=None):
        """Generate the next clean latent from a (possibly corrupted) context.

        ``context_actions`` are the *arriving* actions of the context frames,
        aligned one-to-one with ``ctx_z`` (``ctx_z[t] <-> context_actions[t]``),
        exactly as consumed by :meth:`agentOutputs`.  The caller is responsible
        for passing the same history it gave the agent; this method never
        fabricates, shifts or zero-pads context actions.  Passing a context
        without its actions raises ``ValueError`` (in ``_forward_tokens``).

        ``context_tau_idx`` lets the caller pass a context that has already been
        corrupted by :meth:`prepareContext` together with the matching signal
        index, so ``agentOutputs`` and ``imagineLatent`` condition on the *same*
        history.  When it is ``None`` the context is corrupted here with the
        module default ``tau_ctx`` (keeps the standalone API self-contained).
        """
        B, C, N, D = ctx_z.shape
        if context_tau_idx is None:
            ctx, context_tau_idx = self.prepareContext(ctx_z)
        else:
            ctx = ctx_z
        z = torch.randn(B, 1, N, D, device=ctx_z.device)
        tau = torch.zeros(B, 1, device=ctx_z.device)
        d = 1.0 / K
        d_idx = torch.tensor([0], device=ctx_z.device).expand(B, 1)

        for _ in range(K):
            tau_idx = self._tau_index(tau)
            pred = self.forward(
                z, tau_idx, d_idx, actions, context_z=ctx,
                context_tau_idx=context_tau_idx,
                context_actions=context_actions,
            )
            velocity = (pred - z) / (1.0 - tau[..., None, None]).clamp_min(1e-6)
            z = z + velocity * d
            tau = tau + d
        return z
