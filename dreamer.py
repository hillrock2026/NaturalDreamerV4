"""DreamerV4-style agent (small prototype).

Project positioning: this is a *paper-mechanism-driven* small-scale prototype,
not a line-by-line or structurally-identical port of the official ``dreamer4``
reference implementation.

* NaturalDreamer contributes the training/replay/evaluation/experiment
  skeleton (buffer, checkpoints, CSV/plotly metrics, env wrapper).
* ``tokenizer`` and ``dynamics`` follow the paper plus the official
  ``dreamer4`` code where a reference exists (Phase 1a tokenizer, Phase 1b
  dynamics / shortcut forcing).
* Phase 2 (MTP behaviour cloning + reward) and Phase 3 (imagination + PMPO)
  are paper-driven extensions: the official code does not ship these stages,
  so they are implemented from the paper's equations, not migrated from
  reference code.
 * RoPE (vs additive sinusoidal), QKNorm, soft capping, GQA, ``time_every``,
   ``LossNormalizer`` and ``tauCtx`` are project choices / to-be-validated
   differences from the reference, not verified against it.

Precision contract
------------------
The formal training precision is ``fp32``.  ``trainPhase1a/1b/2/3`` run in full
float32 and never enter a ``torch.autocast`` context.  The config value is
validated by :func:`resolvePrecision`; ``"bf16"`` is rejected fail-fast rather
than silently executed in float32, so the configured precision can never
diverge from the executed one.  The bf16 operator probes in
``tests/test_transformer.py`` are capability checks only and are *not* evidence
that formal bf16 training is supported.
"""

import copy
import os
from collections import deque

import imageio
import numpy as np
import torch
import torch.nn as nn

from buffer import ReplayBuffer
from dynamics import DynamicsTransformer
from heads import PolicyHead, RewardHead, ValueHead, discretize_actions, discrete_to_actions, pmpo_loss
from lossnorm import LossNormalizer
from tokenizer import CausalTokenizer
from utils import atomicWriteFile, computeLambdaValues


def buildTokenizerOptimizer(tokenizer, lr, weight_decay):
    """Build the Phase 1a tokenizer optimizer with decoupled weight decay.

    Task 8B Phase 1a root-cause fix (F1).  The historical
    ``torch.optim.Adam(tokenizer.parameters(), weight_decay=wd)`` adds
    ``wd * p`` to the gradient (L2-in-Adam).  For encoder/latent weight
    matrices whose data gradient is small, the L2 term dominates the adaptive
    normalization and the effective step becomes ``~lr * sign(p)``, shrinking
    them at ``~lr`` per step.  Short-range diagnosis (2000-step ablations D0-D5)
    showed this shrank the encoder weight norms by ~1e3x within 2000 steps and
    decorrelated the latent from the input; switching to decoupled AdamW (D2) or
    removing decay (D1) prevented it.

    LayerNorm parameters, biases and embeddings/token parameters are placed in a
    no-decay group; the remaining weight matrices keep a decoupled (AdamW) decay.
    This only changes the optimizer, never the model graph or state_dict.
    """
    decay, no_decay = [], []
    for name, param in tokenizer.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim <= 1 or name.endswith(".bias") or "norm" in name
                or "embed" in name or name in ("mask_token", "patch_pos")):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr)


def resolvePrecision(precision):
    """Validate the configured training precision and return its canonical name.

    Only ``"fp32"`` is supported for formal training.  ``"bf16"`` is rejected
    with a clear error because the training phases do not run under
    ``torch.autocast`` and no bf16 forward/backward/optimizer path has been
    validated; silently running bf16 configs in float32 would hide that.
    """
    if precision is None:
        precision = "fp32"
    if not isinstance(precision, str):
        raise ValueError(
            f"precision must be a string, got {type(precision).__name__}."
        )
    canonical = precision.strip().lower()
    if canonical == "fp32":
        return "fp32"
    if canonical == "bf16":
        raise ValueError(
            "precision='bf16' is not supported for formal training: the training "
            "phases do not run under torch.autocast and bf16 forward/backward/"
            "optimizer stability has not been validated. Set precision: fp32. "
            "The bf16 operator probes in tests/test_transformer.py are capability "
            "checks only, not a training precision path."
        )
    raise ValueError(
        f"Unknown precision '{precision}'. The only supported formal value is 'fp32'."
    )



class DreamerV4(nn.Module):
    def __init__(self, observation_shape, action_size, action_low, action_high, device, config):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.action_low = list(action_low)
        self.action_high = list(action_high)
        self.config = config
        self.device = device
        self.precision = resolvePrecision(getattr(config, "precision", "fp32"))

        self.image_size = config.imageSize
        self.patch_size = config.patchSize
        self.patch_channels = config.patchChannels
        self.context_length = config.contextLength
        self.imagination_horizon = config.imaginationHorizon
        self.mtp_length = config.mtpLength
        self.action_bins = config.actionBins
        self.twohot_bins = config.twohot.bins
        self.twohot_low = config.twohot.low
        self.twohot_high = config.twohot.high

        self.tokenizer = CausalTokenizer(
            self.image_size, self.patch_size, self.patch_channels, config.tokenizer
        )
        self.dynamics = DynamicsTransformer(
            config.tokenizer.latentTokens,
            config.tokenizer.latentDim,
            action_size,
            config.dynamics,
        )
        # The context corruption level lives at the dreamer level
        # (``dreamer.tauCtx``); forward it to the dynamics module so
        # ``prepareContext`` honours the configuration.
        self.dynamics.tau_ctx = float(getattr(config, "tauCtx", 0.1))

        dynamics_dim = config.dynamics.modelDim
        self.policyHead = PolicyHead(
            dynamics_dim, action_size, self.action_bins, self.mtp_length, config.headHidden
        )
        self.rewardHead = RewardHead(
            dynamics_dim, self.mtp_length, config.headHidden, self.twohot_bins, self.twohot_low, self.twohot_high
        )
        self.valueHead = ValueHead(
            dynamics_dim, config.headHidden, self.twohot_bins, self.twohot_low, self.twohot_high
        )

        self.lossnorms = nn.ModuleDict({
            "tokenizerMSE": LossNormalizer(device, decay=config.lossNormDecay),
            "tokenizerLPIPS": LossNormalizer(device, decay=config.lossNormDecay),
            "flow": LossNormalizer(device, decay=config.lossNormDecay),
            "bc": LossNormalizer(device, decay=config.lossNormDecay),
            "reward": LossNormalizer(device, decay=config.lossNormDecay),
            "value": LossNormalizer(device, decay=config.lossNormDecay),
            "pmpo": LossNormalizer(device, decay=config.lossNormDecay),
        })

        self.tokenizerOptimizer = buildTokenizerOptimizer(
            self.tokenizer, config.lr, config.weightDecay
        )
        self.dynamicsOptimizer = torch.optim.Adam(
            self.dynamics.parameters(), lr=config.lr, weight_decay=config.weightDecay
        )
        self.phase2Optimizer = torch.optim.Adam(
            list(self.dynamics.parameters()) + list(self.policyHead.parameters()) + list(self.rewardHead.parameters()),
            lr=config.phase2Lr,
            weight_decay=config.weightDecay,
        )
        self.phase3Optimizer = torch.optim.Adam(
            list(self.policyHead.parameters()) + list(self.valueHead.parameters()),
            lr=config.lr,
            weight_decay=config.weightDecay,
        )

        self.buffer = ReplayBuffer(observation_shape, action_size, config.buffer, device)
        self.frozenPrior = None
        self.total_episodes = 0
        self.total_env_steps = 0
        self.total_gradient_steps = 0
        self.to(device)

    def freezePolicyPrior(self):
        self.frozenPrior = copy.deepcopy(self.policyHead)
        self.frozenPrior.requires_grad_(False)
        self.frozenPrior.eval()

    def finalizePhase2(self):
        """Create the frozen policy prior exactly once, after Phase 2 ends.

        The prior is a deepcopy of the current (final) policy head, shares no
        parameters with ``policyHead``, has ``requires_grad=False`` and
        ``eval()`` mode, and is not updated by the Phase 3 optimizer.  Called
        explicitly by the training driver, not on every Phase 2 step.
        """
        self.freezePolicyPrior()

    def precisionReport(self):
        """Return the actual runtime precision/dtype state, not the raw config.

        Parameter dtypes are reported as a sorted set so that a partially
        converted model can never be masked by inspecting a single parameter.
        """
        parameters = list(self.parameters())
        model_dtypes = sorted({str(p.dtype) for p in parameters})
        buffer_dtypes = sorted({
            f"numpy.{self.buffer.observations.dtype}",
            f"numpy.{self.buffer.nextObservations.dtype}",
            f"numpy.{self.buffer.actions.dtype}",
            f"numpy.{self.buffer.rewards.dtype}",
            f"numpy.{self.buffer.dones.dtype}",
            f"numpy.{self.buffer.relevant.dtype}",
        })
        device = str(parameters[0].device) if parameters else str(self.device)
        return {
            "precision": self.precision,
            "model_dtypes": model_dtypes,
            "buffer_dtypes": buffer_dtypes,
            "autocast_enabled": bool(torch.is_autocast_enabled()),
            "device": device,
        }

    def logPrecisionContract(self):
        """Print the actual precision/dtype contract once at training start."""
        report = self.precisionReport()
        print(f"precision: {report['precision']}")
        print(f"model_dtype: {', '.join(report['model_dtypes'])}")
        print(f"buffer_dtype: {', '.join(report['buffer_dtypes'])}")
        print(f"autocast_enabled: {'true' if report['autocast_enabled'] else 'false'}")
        print(f"device: {report['device']}")
        if len(report["model_dtypes"]) > 1:
            print(f"WARNING: mixed parameter dtypes detected: {report['model_dtypes']}")
        return report

    @staticmethod
    def _arrivingActions(actions):
        """Convert executing actions to the arriving convention.

        The replay dataset stores the action *executed* at frame ``t``
        (``actions[t]`` produces ``obs[t+1]``).  The world model and the policy
        rollout use the *arriving* convention of the reference implementation:
        frame ``t`` carries the action that produced it, ``slot[t] = a[t-1]``,
        with a dummy ``0`` on the first frame of a window.  Shifting here (rather
        than re-collecting data) keeps the offline dataset and buffer semantics
        intact and matches the online interaction history.
        """
        shifted = torch.zeros_like(actions)
        if actions.shape[1] > 1:
            shifted[:, 1:] = actions[:, :-1]
        return shifted

    def _single_frame_maybe(self, video):
        if torch.rand(1).item() < self.config.tokenizer.singleFrameProb:
            return video[:, :1]
        return video

    @staticmethod
    def _relevance_mask(is_relevant, want_relevant):
        """Return a bool mask of shape (B, T) selecting the wanted group.

        ``is_relevant`` must have shape (B, T, 1).  ``want_relevant=True``
        selects relevant transitions (isRelevant=True), otherwise uniform ones.
        """
        if is_relevant.ndim != 3 or is_relevant.shape[-1] != 1:
            raise ValueError(
                f"isRelevant must have shape (B, T, 1), got {tuple(is_relevant.shape)}."
            )
        mask = is_relevant[..., 0].bool()
        return mask if want_relevant else ~mask

    @staticmethod
    def _masked_mean(per_sample, mask, context):
        """Reduce per-sample losses ``(B, T)`` over ``mask`` (B, T) bool.

        Raises ``ValueError`` if the wanted group is empty instead of
        silently producing NaN or falling back to the wrong group.  This is a
        deliberate fail-fast policy: with a mixed (relevant/uniform) dataset
        the probability of an empty group is negligible, while an all-uniform
        or all-relevant dataset is a data/config error that should surface
        immediately rather than silently train on the wrong subset.  No
        resampling is performed and no NaN can be produced.
        """
        mask = mask.to(per_sample.device).float()
        count = mask.sum()
        if count.item() == 0:
            raise ValueError(
                f"Empty data group for {context}: the batch has no "
                f"{'relevant' if 'relevant' in context else 'uniform'} samples. "
                "Refusing to fall back to the wrong data."
            )
        return (per_sample * mask).sum() / count

    def trainPhase1a(self, batch):
        video = batch.observations
        video = self._single_frame_maybe(video)
        recon, z, mask = self.tokenizer(video, mask_ratio=self.config.tokenizer.maskRatioMax)
        tokenizer_loss, metrics = self.tokenizer.loss(
            video, recon, z, mask,
            {"mse": self.lossnorms["tokenizerMSE"], "lpips": self.lossnorms["tokenizerLPIPS"]},
        )
        self.tokenizerOptimizer.zero_grad()
        tokenizer_loss.backward()
        nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.tokenizerOptimizer.step()
        metrics["psnr"] = -10.0 * torch.log10(torch.tensor(metrics["mse"] + 1e-12)).item()
        return metrics

    def trainPhase1b(self, batch):
        video = batch.observations
        with torch.no_grad():
            z1, _ = self.tokenizer.encode(video)
        # World-model action slots use the arriving convention (slot[t] = a[t-1]).
        arriving_actions = self._arrivingActions(batch.actions)
        per_sample, metrics = self.dynamics.shortcutForcingLoss(z1, arriving_actions)
        # Phase 1b dynamics loss applies to uniform sequences only (isRelevant=False).
        uniform_mask = self._relevance_mask(batch.isRelevant, want_relevant=False)
        flow_loss = self._masked_mean(per_sample, uniform_mask, context="Phase1b dynamics (uniform)")
        flow_loss = self.lossnorms["flow"](flow_loss)
        self.dynamicsOptimizer.zero_grad()
        flow_loss.backward()
        nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.dynamicsOptimizer.step()
        metrics["flowLoss"] = flow_loss.item()
        return metrics

    def trainPhase2(self, batch):
        """Phase 2 objective and gradient boundary.

        Losses (data masks per the design doc):
          * dynamics shortcut forcing on **uniform** sequences only
            (``isRelevant=False``), so the world model does not learn to
            predict optimistically from high-reward segments;
          * MTP behaviour-cloning loss on **relevant** sequences only
            (``isRelevant=True``);
          * MTP reward loss on **relevant** sequences only.

        Updated parameters: dynamics + policyHead + rewardHead (the
        phase2Optimizer).  The tokenizer is frozen (``encode`` runs under
        ``no_grad``) and valueHead is untouched until Phase 3.
        """
        video = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        with torch.no_grad():
            z1, _ = self.tokenizer.encode(video)

        # Dynamics input uses the arriving convention; the MTP/reward *targets*
        # stay executing (distance=0 predicts the action executed at frame t),
        # which is exactly what removes the old same-frame BC leakage.
        arriving_actions = self._arrivingActions(actions)
        per_sample, flow_metrics = self.dynamics.shortcutForcingLoss(z1, arriving_actions)
        uniform_mask = self._relevance_mask(batch.isRelevant, want_relevant=False)
        flow_loss = self._masked_mean(per_sample, uniform_mask, context="Phase2 dynamics (uniform)")
        flow_loss = self.lossnorms["flow"](flow_loss)

        h = self.dynamics.agentOutputs(z1, arriving_actions)
        target_actions = discretize_actions(actions, self.action_low, self.action_high, self.action_bins)

        rel = batch.isRelevant  # (B, T, 1)
        bc_logps, bc_masks, reward_terms, reward_masks = [], [], [], []
        for distance in range(self.mtp_length + 1):
            shift = distance
            end = h.shape[1] - shift
            if end <= 0:
                continue
            h_used = h[:, :end]
            rel_mask = self._relevance_mask(rel[:, shift:shift + end], want_relevant=True)
            policy_dist = self.policyHead(h_used, distance)
            target = target_actions[:, shift:shift + end]
            logp = policy_dist.log_prob(target)
            bc_logps.append(logp)
            bc_masks.append(rel_mask)
            reward_dist = self.rewardHead(h_used, distance)
            reward_target = rewards[:, shift:shift + end].squeeze(-1)
            reward_terms.append(-reward_dist.log_prob(reward_target))
            reward_masks.append(rel_mask)

        bc_raw = self._masked_mean(
            torch.cat([x.reshape(-1) for x in bc_logps]),
            torch.cat([m.reshape(-1) for m in bc_masks]),
            context="Phase2 BC (relevant)",
        )
        reward_raw = self._masked_mean(
            torch.cat([x.reshape(-1) for x in reward_terms]),
            torch.cat([m.reshape(-1) for m in reward_masks]),
            context="Phase2 reward (relevant)",
        )
        bc_loss = self.lossnorms["bc"](-bc_raw)
        reward_loss = self.lossnorms["reward"](reward_raw)
        total = flow_loss + bc_loss + reward_loss

        self.phase2Optimizer.zero_grad()
        total.backward()
        params = list(self.dynamics.parameters()) + list(self.policyHead.parameters()) + list(self.rewardHead.parameters())
        nn.utils.clip_grad_norm_(params, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.phase2Optimizer.step()

        metrics = dict(flow_metrics)
        metrics["flowLoss"] = flow_loss.item()
        metrics["bcLoss"] = bc_loss.item()
        metrics["rewardLoss"] = reward_loss.item()
        metrics["phase2Loss"] = total.item()
        return metrics

    def trainPhase3(self, batch):
        if self.frozenPrior is None:
            raise RuntimeError("Phase 3 requires a frozen policy prior. Run Phase 2 first.")

        video = batch.observations
        actions = batch.actions
        B, T, A = actions.shape
        C = min(self.context_length, T)

        with torch.no_grad():
            ctx_z, _ = self.tokenizer.encode(video[:, :C])

        # Arriving convention throughout the rollout: the history starts with
        # the real context actions shifted by one (slot[t] = a[t-1], dummy 0 at
        # frame 0), and every newly generated frame carries the action that
        # produced it.
        hist_z = ctx_z
        hist_act = self._arrivingActions(actions[:, :C])
        h_list, action_idx_list, reward_list, value_list = [], [], [], []

        # Phase 3 must not run the frozen world model in training mode: the
        # dynamics attention dropout would otherwise inject noise into every
        # imagined step.  The policy/value heads stay in training mode.
        tokenizer_was_training = self.tokenizer.training
        dynamics_was_training = self.dynamics.training
        frozen_prior_was_training = self.frozenPrior.training
        self.tokenizer.eval()
        self.dynamics.eval()
        self.frozenPrior.eval()
        try:
            for _ in range(self.imagination_horizon):
                # Corrupt the history once so the agent hidden state and the
                # generated next frame condition on exactly the same context.
                context_corrupt, context_tau = self.dynamics.prepareContext(hist_z)
                current_z = hist_z[:, -1:]
                current_act = hist_act[:, -1:]
                context_z_used = context_corrupt[:, :-1]
                context_act = hist_act[:, :-1]

                with torch.no_grad():
                    h = self.dynamics.agentOutputs(
                        current_z,
                        current_act,
                        context_z=context_z_used,
                        context_tau_idx=context_tau,
                        context_actions=context_act,
                    )
                h_last = h[:, -1:]

                policy_dist = self.policyHead(h_last, 0)
                action_idx = policy_dist.sample()
                action = discrete_to_actions(
                    action_idx, self.action_low, self.action_high, self.action_bins
                ).detach()

                with torch.no_grad():
                    z_next = self.dynamics.imagineLatent(
                        context_corrupt,
                        action,
                        K=self.config.sampleSteps,
                        context_tau_idx=context_tau,
                        context_actions=hist_act,
                    )

                reward_pred = self.rewardHead(h_last, 0).mean.detach()
                value_pred = self.valueHead(h_last).mean.detach()

                h_list.append(h_last.detach())
                action_idx_list.append(action_idx)
                reward_list.append(reward_pred)
                value_list.append(value_pred)

                hist_z = torch.cat((hist_z, z_next), dim=1)
                hist_act = torch.cat((hist_act, action), dim=1)
        finally:
            if tokenizer_was_training:
                self.tokenizer.train()
            if dynamics_was_training:
                self.dynamics.train()
            if frozen_prior_was_training:
                self.frozenPrior.train()

        h_imag = torch.cat(h_list, dim=1)
        action_indices = torch.cat(action_idx_list, dim=1)
        rewards_imag = torch.cat(reward_list, dim=1).squeeze(-1)
        values_imag = torch.cat(value_list, dim=1).squeeze(-1)

        reward_targets = rewards_imag[:, :-1]
        values_all = values_imag
        continues = torch.full_like(reward_targets, self.config.discount)
        lambda_values = computeLambdaValues(
            reward_targets, values_all, continues, self.config.lambda_
        )
        advantages = lambda_values - values_imag[:, :-1]

        policy_dist = self.policyHead(h_imag[:, :-1], 0)
        logp = policy_dist.log_prob(action_indices[:, :-1])
        prior_dist = self.frozenPrior(h_imag[:, :-1], 0)

        pmpo = pmpo_loss(
            logp,
            policy_dist,
            prior_dist,
            advantages,
            alpha=self.config.pmpoAlpha,
            beta=self.config.pmpoBeta,
        )
        value_loss = self.lossnorms["value"](
            -self.valueHead(h_imag[:, :-1]).log_prob(lambda_values.detach()).mean()
        )
        pmpo_loss_value = self.lossnorms["pmpo"](pmpo)
        total = pmpo_loss_value + value_loss

        self.phase3Optimizer.zero_grad()
        total.backward()
        params = list(self.policyHead.parameters()) + list(self.valueHead.parameters())
        nn.utils.clip_grad_norm_(params, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.phase3Optimizer.step()

        return {
            "phase3Loss": total.item(),
            "pmpoloss": pmpo.item(),
            "valueLoss": value_loss.item(),
            "advantages": advantages.mean().item(),
        }

    @torch.no_grad()
    def imagine(self, ctx_z, actions, context_actions=None):
        """Generate the next latent from ``ctx_z`` and its arriving actions.

        ``context_actions[t]`` must be the arriving action of context frame
        ``t`` (same convention as ``agentOutputs``); omitting it for a
        non-empty context is a hard error (no silent zero fallback).
        """
        return self.dynamics.imagineLatent(
            ctx_z, actions, K=self.config.sampleSteps, context_actions=context_actions
        )

    @torch.no_grad()
    def environmentInteraction(
        self,
        env,
        num_episodes,
        seed=None,
        evaluation=False,
        save_video=False,
        filename="videos/unnamedVideo",
        fps=30,
        macro_block_size=16,
        relevant_episode_quantile=0.75,
    ):
        """Run episodes, then write them to the buffer with per-episode relevance.

        Episodes are buffered in full and written only after every episode in
        the call has finished.  The top ``1 - relevant_episode_quantile``
        fraction by total return is marked ``relevant=True`` (the Phase 2
        behaviour-cloning / reward group); the rest is ``relevant=False`` (the
        world-model / uniform group).  The count is clamped so at least one
        episode stays in each group, and at least two episodes are required.

        ``done == 1`` is stored on the last transition of every episode and is
        used by ``ReplayBuffer`` as an episode boundary, so no sampled sequence
        can cross it.  With ``evaluation=True`` nothing is written to the
        training buffer and no relevance marking happens.
        """
        if not evaluation:
            if not 0.0 <= relevant_episode_quantile < 1.0:
                raise ValueError(
                    "relevant_episode_quantile must be in [0.0, 1.0), got "
                    f"{relevant_episode_quantile}."
                )
            if num_episodes < 2:
                raise ValueError(
                    "Online relevance marking needs at least 2 warmup episodes "
                    "so that both the relevant and uniform groups are non-empty; "
                    f"got num_episodes={num_episodes}. Increase episodesBeforeStart."
                )

        scores = []
        pending_episodes = []
        for episode in range(num_episodes):
            observation = env.reset(seed=(seed + self.total_episodes if seed else None))
            history = deque(maxlen=self.context_length)
            action_history = deque(maxlen=self.context_length)
            history.append(observation)
            action_history.append(np.zeros(self.action_size, dtype=np.float32))
            score, done, frames = 0.0, False, []
            transitions = []

            while not done:
                if len(history) >= self.context_length:
                    obs = np.stack(history, axis=0)
                    video = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    z, _ = self.tokenizer.encode(video)
                    acts = torch.from_numpy(np.stack(action_history, axis=0)).float().unsqueeze(0).to(self.device)
                    h = self.dynamics.agentOutputs(z, acts)
                    dist = self.policyHead(h[:, -1], 0)
                    action_idx = dist.sample()
                    action = discrete_to_actions(action_idx, self.action_low, self.action_high, self.action_bins)
                    action = action.cpu().numpy().reshape(-1)
                else:
                    action = env.action_space.sample()

                next_observation, reward, done = env.step(action)
                if not evaluation:
                    transitions.append((observation, action, reward, next_observation, done))

                if save_video and episode == 0:
                    frame = env.render()
                    target_h = (frame.shape[0] + macro_block_size - 1) // macro_block_size * macro_block_size
                    target_w = (frame.shape[1] + macro_block_size - 1) // macro_block_size * macro_block_size
                    frames.append(np.pad(frame, ((0, target_h - frame.shape[0]), (0, target_w - frame.shape[1]), (0, 0)), mode="edge"))

                history.append(next_observation)
                action_history.append(action)
                observation = next_observation
                score += reward
                if done:
                    scores.append(score)
                    if not evaluation:
                        pending_episodes.append((score, transitions))
                        self.total_episodes += 1
                        self.total_env_steps += len(transitions)
                    if save_video and episode == 0:
                        final_filename = f"{filename}_reward_{score:.0f}.mp4"
                        with imageio.get_writer(final_filename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break
        if not evaluation and pending_episodes:
            scores_array = np.asarray(
                [score for score, _ in pending_episodes], dtype=np.float32
            )
            relevant_count = max(
                1,
                min(
                    len(pending_episodes) - 1,
                    int(np.ceil((1.0 - relevant_episode_quantile) * len(pending_episodes))),
                ),
            )
            ranked_indices = np.argsort(scores_array, kind="stable")
            relevant_indices = set(ranked_indices[-relevant_count:].tolist())
            for episode_index, (_, transitions) in enumerate(pending_episodes):
                relevant = episode_index in relevant_indices
                for transition in transitions:
                    self.buffer.add(*transition, relevant=relevant)
        return sum(scores) / num_episodes if num_episodes else None

    def _checkpointPayload(self):
        """Build the checkpoint dict that is persisted by save methods.

        The field set and semantics are unchanged; only the way it is written
        to disk (atomic temp file + fsync + replace) has changed.
        """
        return {
            "tokenizer": self.tokenizer.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "policyHead": self.policyHead.state_dict(),
            "rewardHead": self.rewardHead.state_dict(),
            "valueHead": self.valueHead.state_dict(),
            "frozenPrior": None if self.frozenPrior is None else self.frozenPrior.state_dict(),
            "tokenizerOptimizer": self.tokenizerOptimizer.state_dict(),
            "dynamicsOptimizer": self.dynamicsOptimizer.state_dict(),
            "phase2Optimizer": self.phase2Optimizer.state_dict(),
            "phase3Optimizer": self.phase3Optimizer.state_dict(),
            "totalEpisodes": self.total_episodes,
            "totalEnvSteps": self.total_env_steps,
            "totalGradientSteps": self.total_gradient_steps,
        }

    @staticmethod
    def _normalizeCheckpointPath(path):
        if not path.endswith(".pth"):
            path += ".pth"
        return path

    def saveCheckpoint(self, path):
        """Atomically save a training-time checkpoint, allowing overwrite.

        Regular checkpoints keep their historical overwrite semantics (a
        resumed run may re-save the same step suffix); only the final
        checkpoint uses the no-overwrite ``saveFinalCheckpoint`` entry point.
        """
        path = self._normalizeCheckpointPath(path)
        checkpoint = self._checkpointPayload()
        atomicWriteFile(path, lambda handle: torch.save(checkpoint, handle), overwrite=True)

    def saveFinalCheckpoint(self, path):
        """Atomically publish a final checkpoint without overwriting.

        Unlike ``saveCheckpoint``, an existing target is never replaced: the
        call fails before writing anything so a partially-trusted or historical
        final checkpoint cannot be silently clobbered.
        """
        path = self._normalizeCheckpointPath(path)
        checkpoint = self._checkpointPayload()
        atomicWriteFile(path, lambda handle: torch.save(checkpoint, handle), overwrite=False)

    def loadCheckpoint(self, path):
        path = self._normalizeCheckpointPath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.tokenizer.load_state_dict(checkpoint["tokenizer"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.policyHead.load_state_dict(checkpoint["policyHead"])
        self.rewardHead.load_state_dict(checkpoint["rewardHead"])
        self.valueHead.load_state_dict(checkpoint["valueHead"])
        if checkpoint.get("frozenPrior") is not None:
            self.freezePolicyPrior()
            self.frozenPrior.load_state_dict(checkpoint["frozenPrior"])
        else:
            # The checkpoint is the source of truth: a checkpoint without a
            # frozen prior (e.g. phase1a/1b) must not leave a stale prior from a
            # previous load in place.
            self.frozenPrior = None
        self.tokenizerOptimizer.load_state_dict(checkpoint["tokenizerOptimizer"])
        self.dynamicsOptimizer.load_state_dict(checkpoint["dynamicsOptimizer"])
        self.phase2Optimizer.load_state_dict(checkpoint["phase2Optimizer"])
        self.phase3Optimizer.load_state_dict(checkpoint["phase3Optimizer"])
        self.total_episodes = checkpoint["totalEpisodes"]
        self.total_env_steps = checkpoint["totalEnvSteps"]
        self.total_gradient_steps = checkpoint["totalGradientSteps"]
