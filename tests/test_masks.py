import torch

from masks import (
    make_block_causal_mask,
    make_dynamics_mask,
    make_space_only_mask,
    make_time_only_mask,
    make_tokenizer_encoder_mask,
)


def test_block_causal_shape():
    T, S = 3, 5
    mask = make_block_causal_mask(T, S, "cpu")
    assert mask.shape == (T * S, T * S)
    q = torch.arange(T * S)
    k = torch.arange(T * S)
    q_t = q // S
    k_t = k // S
    assert torch.equal(mask, torch.where(k_t[None, :] <= q_t[:, None], torch.zeros(T * S, T * S), torch.full((T * S, T * S), float("-inf"))))


def test_space_only_diagonal_blocks():
    T, S = 3, 5
    mask = make_space_only_mask(T, S, "cpu")
    for q in range(T * S):
        for k in range(T * S):
            if q // S == k // S:
                assert mask[q, k] == 0.0
            else:
                assert mask[q, k] == float("-inf")


def test_time_only_same_position():
    T, S = 3, 5
    mask = make_time_only_mask(T, S, "cpu")
    for q in range(T * S):
        for k in range(T * S):
            same_position = (q % S) == (k % S)
            past_or_same = (k // S) <= (q // S)
            expected = 0.0 if (same_position and past_or_same) else float("-inf")
            assert mask[q, k] == expected


def test_dynamics_agent_asymmetry():
    T, S, S_agent = 3, 5, 2
    mask = make_dynamics_mask(T, S, S_agent, "cpu")
    total_S = S + S_agent
    for t in range(T):
        for i in range(total_S):
            is_agent_i = i >= S
            for tp in range(T):
                for j in range(total_S):
                    is_agent_j = j >= S
                    val = mask[t * total_S + i, tp * total_S + j]
                    if tp <= t:
                        if is_agent_i or not is_agent_j:
                            assert val == 0.0, f"({t},{i}) should see ({tp},{j})"
                        else:
                            assert val == float("-inf"), f"non-agent ({t},{i}) must NOT see agent ({tp},{j})"
                    else:
                        assert val == float("-inf"), "future blocked"


def test_tokenizer_encoder_modality():
    T, S_patch, S_latent = 3, 4, 2
    mask = make_tokenizer_encoder_mask(T, S_patch, S_latent, "cpu")
    total = S_patch + S_latent
    for q in range(T * total):
        q_t, q_pos = q // total, q % total
        q_is_patch = q_pos < S_patch
        for k in range(T * total):
            k_t, k_pos = k // total, k % total
            k_is_patch = k_pos < S_patch
            expected = 0.0 if (k_t <= q_t and (not q_is_patch or k_is_patch)) else float("-inf")
            assert mask[q, k] == expected


def test_all_masks_no_infs_on_diagonal():
    T, S, S_agent = 3, 5, 2
    masks = [
        make_block_causal_mask(T, S, "cpu"),
        make_space_only_mask(T, S, "cpu"),
        make_time_only_mask(T, S, "cpu"),
        make_dynamics_mask(T, S, S_agent, "cpu"),
        make_tokenizer_encoder_mask(T, S, S_agent, "cpu"),
    ]
    for mask in masks:
        assert torch.all(torch.diagonal(mask) == 0.0)
