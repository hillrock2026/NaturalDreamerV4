import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


class QKNorm(nn.Module):
    """L2-normalize Q and K before attention."""

    def __init__(self, head_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps) * self.scale


def apply_soft_capping(logits, cap=50.0):
    return cap * torch.tanh(logits / cap)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class RoPE2D(nn.Module):
    """Simplified 2D RoPE: time rotary on one half, space rotary on the other."""

    def __init__(self, head_dim, max_T=512, max_S=512, base=10000.0, device="cpu"):
        super().__init__()
        if head_dim <= 0 or head_dim % 4 != 0:
            raise ValueError(
                "RoPE2D requires head_dim to be a positive multiple of 4 "
                f"(got {head_dim}). The feature axis is split into a rotate_half "
                "pair (head_dim // 2) and each half into time/space quarters "
                "(head_dim // 4); other values yield an empty or mis-sized "
                "frequency tensor instead of a valid rotation."
            )
        self.head_dim = head_dim
        self.max_T = max_T
        self.max_S = max_S
        half = head_dim // 2
        dim = half // 2
        freqs = 1.0 / (base ** (torch.arange(0, dim, device=device).float() / dim))
        pos_t = torch.arange(max_T, device=device).float()
        pos_s = torch.arange(max_S, device=device).float()
        self.register_buffer("cos_t", torch.cos(torch.outer(pos_t, freqs)))
        self.register_buffer("sin_t", torch.sin(torch.outer(pos_t, freqs)))
        self.register_buffer("cos_s", torch.cos(torch.outer(pos_s, freqs)))
        self.register_buffer("sin_s", torch.sin(torch.outer(pos_s, freqs)))

    def forward(self, q, k, T, S):
        B, H, L, D = q.shape
        if L != T * S:
            raise ValueError(
                f"RoPE2D expects L == T * S (got L={L}, T={T}, S={S}). "
                "Positions are derived from a row-major (T, S) layout via "
                "idx // S and idx % S, so a mismatched sequence length has no "
                "defined time/space mapping."
            )
        if T > self.max_T:
            raise ValueError(
                f"RoPE2D was built with max_T={self.max_T} but received T={T}."
            )
        if S > self.max_S:
            raise ValueError(
                f"RoPE2D was built with max_S={self.max_S} but received S={S}."
            )
        q_idx = torch.arange(L, device=q.device)
        steps = q_idx // S
        pos = q_idx % S

        cos_t = self.cos_t[steps]  # (L, D/4)
        sin_t = self.sin_t[steps]
        cos_s = self.cos_s[pos]
        sin_s = self.sin_s[pos]

        cos = torch.cat((cos_t, cos_s), dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.cat((sin_t, sin_s), dim=-1).unsqueeze(0).unsqueeze(0)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads=None, soft_cap=50.0, dropout=0.0):
        super().__init__()
        if heads <= 0 or dim % heads != 0:
            raise ValueError(
                "Attention requires heads to be a positive divisor of dim "
                f"(got dim={dim}, heads={heads}); otherwise head_dim=dim//heads "
                "silently drops the remainder and changes the model dimension."
            )
        self.heads = heads
        self.kv_heads = kv_heads or heads
        self.head_dim = dim // heads
        self.soft_cap = soft_cap

        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.qknorm = QKNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, x, mask=None, rope=None, T=None, S=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)

        q = self.qknorm(q)
        k = self.qknorm(k)

        if rope is not None:
            if T is None or S is None:
                T, S = L, 1
            q, k = rope(q, k, T, S)

        if self.heads != self.kv_heads:
            rep = self.heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        logits = apply_soft_capping(logits, self.soft_cap)

        if mask is not None:
            logits = logits + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(logits, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, kv_heads, mlp_ratio=4, kind="space", soft_cap=50.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, heads, kv_heads, soft_cap, dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio)
        self.kind = kind

    def forward(self, x, T, S, space_mask=None, time_mask=None, rope_space=None, rope_time=None):
        if self.kind == "space":
            mask, rope = space_mask, rope_space
        else:
            mask, rope = time_mask, rope_time
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope, T=T, S=S)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBackbone(nn.Module):
    """A small stack of space/time Blocks with a 2D RoPE helper.

    Gradient checkpointing
    ----------------------
    The attention implementation materializes an ``(B, H, L, L)`` logits
    tensor.  For the formal long sequence (``L = T * S_total`` in the
    thousands) retaining those tensors for every block's backward pass exceeds
    a 48 GB GPU.  When ``gradient_checkpointing`` is enabled (the default) and
    the backbone is training with autograd enabled, each ``Block`` is wrapped
    in a non-reentrant ``torch.utils.checkpoint`` region.  Only the block input
    is retained; the forward is recomputed during backward, so the peak memory
    is one block's transient activations instead of all blocks combined.  The
    forward math (soft capping, masks, GQA, RoPE, dropout) is unchanged, and
    the RNG state is preserved so dropout recomputation is identical.
    """

    def __init__(
        self,
        dim,
        layers,
        heads,
        kv_heads,
        mlp_ratio=4,
        time_every=4,
        soft_cap=50.0,
        dropout=0.0,
        max_T=512,
        max_S=512,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.time_every = time_every
        self.gradient_checkpointing = bool(gradient_checkpointing)
        kinds = ["time" if (i + 1) % time_every == 0 else "space" for i in range(layers)]
        self.blocks = nn.ModuleList(
            [Block(dim, heads, kv_heads, mlp_ratio, kind, soft_cap, dropout) for kind in kinds]
        )
        self.rope_space = RoPE2D(dim // heads, max_T=max_T, max_S=max_S)
        self.rope_time = RoPE2D(dim // heads, max_T=max_T, max_S=max_S)

    @staticmethod
    def _run_block(block, x, T, S, space_mask, time_mask, rope_space, rope_time):
        return block(x, T, S, space_mask, time_mask, rope_space, rope_time)

    def forward(self, x, T, S):
        space_mask = None
        time_mask = None
        use_checkpoint = (
            self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        )
        for block in self.blocks:
            if block.kind == "space" and space_mask is None:
                from masks import make_space_only_mask
                space_mask = make_space_only_mask(T, S, x.device)
            elif block.kind == "time" and time_mask is None:
                from masks import make_time_only_mask
                time_mask = make_time_only_mask(T, S, x.device)
            if use_checkpoint:
                x = _checkpoint(
                    self._run_block,
                    block,
                    x,
                    T,
                    S,
                    space_mask,
                    time_mask,
                    self.rope_space,
                    self.rope_time,
                    use_reentrant=False,
                )
            else:
                x = block(x, T, S, space_mask, time_mask, self.rope_space, self.rope_time)
        return x
