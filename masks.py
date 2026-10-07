import torch


def _to_mask(cond):
    return torch.where(cond, torch.zeros_like(cond, dtype=torch.float32), torch.full_like(cond, float("-inf"), dtype=torch.float32))


def make_block_causal_mask(T, S, device):
    """Each token at step t can see every token at steps <= t."""
    L = T * S
    q_idx = torch.arange(L, device=device)
    k_idx = torch.arange(L, device=device)
    q_t = q_idx // S
    k_t = k_idx // S
    return _to_mask(k_t[None, :] <= q_t[:, None])


def make_space_only_mask(T, S, device):
    """Tokens only attend to tokens within the same frame."""
    L = T * S
    q_idx = torch.arange(L, device=device)
    k_idx = torch.arange(L, device=device)
    q_t = q_idx // S
    k_t = k_idx // S
    return _to_mask(k_t[None, :] == q_t[:, None])


def make_time_only_mask(T, S, device):
    """Tokens attend to the same intra-frame position at earlier/current steps."""
    L = T * S
    q_idx = torch.arange(L, device=device)
    k_idx = torch.arange(L, device=device)
    q_t = q_idx // S
    q_s = q_idx % S
    k_t = k_idx // S
    k_s = k_idx % S
    return _to_mask((k_t[None, :] <= q_t[:, None]) & (k_s[None, :] == q_s[:, None]))


def make_dynamics_mask(T, S, S_agent, device):
    """Block-causal mask with asymmetric agent-token visibility.

    Agent tokens may attend to everything up to the current step.  Other
    tokens may not attend to agent tokens, preventing causal confusion.
    """
    total_S = S + S_agent
    L = T * total_S
    q_idx = torch.arange(L, device=device)
    k_idx = torch.arange(L, device=device)
    q_t = q_idx // total_S
    q_pos = q_idx % total_S
    k_t = k_idx // total_S
    k_pos = k_idx % total_S

    q_is_agent = q_pos >= S
    k_is_agent = k_pos >= S
    past_or_same = k_t[None, :] <= q_t[:, None]
    can_see = q_is_agent | (~k_is_agent)
    return _to_mask(past_or_same & (q_is_agent[:, None] | (~k_is_agent)[None, :]))


def make_tokenizer_encoder_mask(T, S_patch, S_latent, device):
    """Patch tokens see patch tokens; latent tokens see all modalities."""
    total_S = S_patch + S_latent
    L = T * total_S
    q_idx = torch.arange(L, device=device)
    k_idx = torch.arange(L, device=device)
    q_t = q_idx // total_S
    q_pos = q_idx % total_S
    k_t = k_idx // total_S
    k_pos = k_idx % total_S

    q_is_patch = q_pos < S_patch
    k_is_patch = k_pos < S_patch
    past_or_same = k_t[None, :] <= q_t[:, None]
    modality_allowed = (~q_is_patch) | k_is_patch
    return _to_mask(past_or_same & ((~q_is_patch)[:, None] | k_is_patch[None, :]))
