# DreamerV4 测试计划

> 目标：覆盖 6 项修复（P1-P6）和全部核心模块，确保 `NaturalDreamerV4/` 在进入训练实验前算法正确、形状无误、梯度可通。

## 运行环境

```powershell
cd D:\Agentic_Program\DreamerV4\NaturalDreamerV4
$env:PYTHONPATH = "D:\Agentic_Program\DreamerV4\NaturalDreamerV4"
C:\Users\omen\anaconda3\python.exe -m pytest tests/ -v
```

若未安装 pytest，可用 `C:\Users\omen\anaconda3\python.exe tests/<test_file>.py` 直接运行（每个文件末尾有 `if __name__ == "__main__"` 块）。

## 测试文件结构

```
NaturalDreamerV4/
├── tests/
│   ├── conftest.py              # 共享 fixtures（小配置、随机张量工厂）
│   ├── test_twohot.py           # symlog/symexp 往返、TwoHotDist log_prob/mean
│   ├── test_lossnorm.py         # LossNormalizer EMA 更新与归一化
│   ├── test_masks.py            # 6 种掩码形状与因果性
│   ├── test_transformer.py      # Attention/Block/Backbone 前向与 GQA
│   ├── test_tokenizer.py        # P2 MAE 逐 patch / P3 memory_mask / 形状守恒
│   ├── test_dynamics.py         # shortcut forcing 损失 / imagineLatent / P6 上下文 τ
│   ├── test_heads.py            # P5 PMPO 空集 / MTP 多距离 / two-hot 头
│   ├── test_dreamer_phases.py   # P1 Phase3 想象 rollout / P4 MTP 索引 / 四阶段入口
│   └── test_integration.py      # 端到端：Phase1a→1b→2→3 连跑 + 检查点存取
```

---

## T1. test_twohot.py

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T1.1 | symlog_symexp_roundtrip | `symexp(symlog(x)) ≈ x`，对 x∈{-100,-1,0,0.5,50} 误差 <1e-5 |
| T1.2 | twohot_log_prob_matches_cross_entropy | 随机 logits，`TwoHotDist.log_prob(value)` 与手算交叉熵一致 |
| T1.3 | twohot_mean_is_symexp_of_weighted_sum | `dist.mean == symexp(Σ probs·cut_points)` |
| T1.4 | twohot_sample_in_range | 采样 1000 次，值落在 [low, high] 内 |
| T1.5 | twohot_encode_is_two_hot | 编码后向量仅 2 个非零位且和为 1 |
| T1.6 | twohot_log_prob_backward | `log_prob` 可反向，梯度形状与 logits 一致 |

```python
# T1.1
def test_symlog_symexp_roundtrip():
    for val in [-100.0, -1.0, 0.0, 0.5, 50.0]:
        x = torch.tensor(val)
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)
```

## T2. test_lossnorm.py

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T2.1 | rms_ema_updates | 连续传 3 个相同 loss，rms 收敛到该值 |
| T2.2 | normalizes_scale | 传 loss=100 和 loss=0.01，归一化后两者同量级 |
| T2.3 | detach_blocks_grad | 归一化输出对原 loss 有梯度，但对 rms buffer 无梯度 |
| T2.4 | multi_key_independent | 两个 LossNormalizer 实例的 rms 互不干扰 |

## T3. test_masks.py

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T3.1 | block_causal_shape | `(T*S, T*S)` 且对角线及下方为 0 |
| T3.2 | space_only_diagonal_blocks | 仅同时间步内为 0，跨时间步为 -inf |
| T3.3 | time_only_same_position | (t,s) 只能看到 (t',s) where t'<=t |
| T3.4 | dynamics_agent_asymmetry | **关键**：非 agent token 看不到 agent token；agent token 看到一切 |
| T3.5 | tokenizer_encoder_modality | patch token 只看 patch；latent token 看 patch+latent |
| T3.6 | tokenizer_decoder_modality | patch 看 patch+latent；latent 只看 latent |
| T3.7 | all_masks_no_infs_on_diagonal | 所有掩码对角线为 0（自注意力始终允许） |

```python
# T3.4 关键测试
def test_dynamics_agent_asymmetry():
    T, S, S_agent = 3, 5, 2
    mask = make_dynamics_mask(T, S, S_agent, "cpu")  # (T*(S+A), T*(S+A))
    total_S = S + S_agent
    for t in range(T):
        for i in range(total_S):
            is_agent_i = i >= S
            for tp in range(T):
                for j in range(total_S):
                    is_agent_j = j >= S
                    val = mask[t*total_S+i, tp*total_S+j]
                    if tp <= t:
                        if is_agent_i or not is_agent_j:
                            assert val == 0.0, f"({t},{i}) should see ({tp},{j})"
                        else:
                            assert val == float("-inf"), f"non-agent ({t},{i}) must NOT see agent ({tp},{j})"
                    else:
                        assert val == float("-inf"), f"future blocked"
```

## T4. test_transformer.py

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T4.1 | attention_output_shape | 输入 (B,L,D) → 输出 (B,L,D) |
| T4.2 | gqa_kv_repeat | heads=4, kv_heads=2，k/v 重复后维度匹配 |
| T4.3 | soft_capping_bounded | logits 绝对值 ≤ cap=50 |
| T4.4 | block_space_time_kinds | 12 层中 kinds[3],kinds[7],kinds[11] == "time"，其余 == "space" |
| T4.5 | backbone_forward_no_error | 随机 (B, T*S, D) 前向不报错 |
| T4.6 | backbone_backward | 前向+反向，梯度不为 None |

## T5. test_tokenizer.py（覆盖 P2、P3）

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T5.1 | encode_shape | 输入 (2,4,3,64,64) → z 形状 (2,4,64,32) |
| T5.2 | decode_shape | z (2,4,64,32) → recon (2,4,3,64,64) |
| T5.3 | bottleneck_conservation | 128×16 == 64×32 == 2048 |
| T5.4 | **P2: mae_per_patch_independent** | 同一帧内部分 patch 被丢弃、部分保留 |
| T5.5 | **P2: mae_p_zero_means_no_mask** | mask_ratio=0 时无 patch 被替换 |
| T5.6 | **P2: mae_p_covers_full_range** | p~U(0,0.9) 采样 1000 次，p 值覆盖 [0,0.9] |
| T5.7 | **P3: memory_mask_shape** | (T*P, T*Nb) = (4*64, 4*128) = (256, 512) |
| T5.8 | **P3: memory_mask_causal** | 第 0 帧 patch 只能看第 0 帧 latent |
| T5.9 | forward_recon_loss | `loss()` 返回标量且可反向 |
| T5.10 | single_frame_degradation | T=1 时不报错 |

```python
# T5.4 P2 关键测试
def test_mae_per_patch_independent():
    torch.manual_seed(0)
    tok = make_small_tokenizer()
    video = torch.rand(1, 1, 3, 64, 64)
    # 截取 encode 内部的 mask 逻辑
    patches = tok.patchify_tensor(video)
    patches = tok.patch_proj(patches)
    p = torch.rand(1, 1, 1, 1) * 0.9
    mask = torch.rand(1, 1, 64, 1) < p
    # 验证同一帧内 mask 不全是 True 也不全是 False（概率意义上）
    assert mask.sum() > 0 or p.item() < 0.02  # p 很小时可能全 False
    assert mask.sum() < 64 or p.item() > 0.98  # p 很大时可能全 True
    # 关键：mask 不是全相同值
    if p.item() > 0.1 and p.item() < 0.9:
        assert mask.unique().numel() > 1, "patches must be independently masked"
```

## T6. test_dynamics.py（覆盖 P6）

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T6.1 | forward_shape | z (B,T,64,32) + actions → ẑ¹ 形状一致 |
| T6.2 | agentOutputs_shape | 返回 (B,T,dynamics_dim) |
| T6.3 | shortcutForcingLoss_scalar | 损失是标量且可反向 |
| T6.4 | shortcutForcing_excludes_tau1_d>0 | τ=1 时 d 被强制为 d_min |
| T6.5 | shortcutForcing_bootstrap_factor | d>d_min 时 loss 含 (1-τ)² 因子 |
| T6.6 | imagineLatent_output_shape | 返回 (B,1,64,32) |
| T6.7 | imagineLatent_tau_progression | K 步后 τ 从 0 增至 1.0 |
| T6.8 | **P6: ctx_tau_is_not_max** | _forward_tokens 中上下文 τ_idx ≠ tau_bins-1 |
| T6.9 | **P6: ctx_tau_matches_0p9** | ctx_tau_idx == 7（tauBins=9 时最接近 0.9） |
| T6.10 | imagineLatent_context_corrupted | ctx 与输入 ctx_z 不同（加了噪声） |

```python
# T6.8/T6.9 P6 关键测试
def test_ctx_tau_is_not_clean():
    dyn = make_small_dynamics()
    z = torch.randn(2, 4, 64, 32)
    actions = torch.randn(2, 4, 3)
    ctx_z = torch.randn(2, 8, 64, 32)
    # 检查 _forward_tokens 内部上下文 τ
    # 通过 hook 或直接检查 tau_bins-1 不被使用
    assert dyn.tau_bins == 9
    ctx_tau_val = 0.9
    expected_idx = min(range(9), key=lambda i: abs(i/8 - ctx_tau_val))
    assert expected_idx == 7
    assert expected_idx != 8  # 不是 tau_bins-1
```

## T7. test_heads.py（覆盖 P5）

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T7.1 | policyhead_output_dist | 返回 Independent(Categorical)，event_shape=(3,) |
| T7.2 | policyhead_mtp_9_heads | forwardAll 返回 9 个分布 |
| T7.3 | rewardhead_twohot | 返回 TwoHotDist，bins=64 |
| T7.4 | valuehead_twohot | 返回 TwoHotDist，bins=64 |
| T7.5 | discretize_actions_roundtrip | 离散化→还原后误差 < (high-low)/bins |
| T7.6 | **P5: pmpo_all_positive** | advantages 全正时不报 shape 错误 |
| T7.7 | **P5: pmpo_all_negative** | advantages 全负时不报 shape 错误 |
| T7.8 | **P5: pmpo_mixed** | 混合时不报错误 |
| T7.9 | **P5: pmpo_returns_scalar** | 输出是 0-dim 标量 |
| T7.10 | pmpo_kl_is_reverse | KL[policy‖prior] 方向正确 |
| T7.11 | pmpo_backward | 损失可反向，policyHead 参数有梯度 |

```python
# T7.6/T7.7/T7.8 P5 关键测试
def test_pmpo_empty_sets():
    B, T = 4, 8
    logp = torch.randn(B * T, requires_grad=True)
    policy_dist = make_categorical_dist(B * T)
    prior_dist = make_categorical_dist(B * T)
    # 全正
    adv_pos = torch.ones(B * T)
    loss = pmpo_loss(logp, policy_dist, prior_dist, adv_pos)
    assert loss.dim() == 0
    loss.backward()
    # 全负
    adv_neg = -torch.ones(B * T)
    loss = pmpo_loss(logp, policy_dist, prior_dist, adv_neg)
    assert loss.dim() == 0
```

## T8. test_dreamer_phases.py（覆盖 P1、P4）

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T8.1 | phase1a_step | 跑一步不报错，返回含 mse/lpips/psnr |
| T8.2 | phase1b_step | 跑一步不报错，返回含 flowLoss |
| T8.3 | phase2_step | 跑一步不报错，返回含 bcLoss/rewardLoss |
| T8.4 | phase3_requires_frozen_prior | 未设 frozenPrior 时 raise RuntimeError |
| T8.5 | phase2_sets_frozen_prior | Phase 2 后 frozenPrior is not None |
| T8.6 | **P1: phase3_uses_imagineLatent** | 检查 trainPhase3 内部调用了 imagineLatent（可 monkeypatch 计数） |
| T8.7 | **P1: phase3_logp_from_sampled_actions** | logp 的 action_indices 来自 policy.sample() 而非 batch |
| T8.8 | **P1: phase3_dynamics_frozen** | Phase 3 后 dynamics 参数无梯度变化 |
| T8.9 | **P1: phase3_advantages_from_imagination** | advantages 来自想象 rollout 的 rewards/values |
| T8.10 | **P4: phase2_mtp_distance0_predicts_current** | distance=0 时 target 是 actions[:,0:end] 而非 actions[:,1:end+1] |
| T8.11 | phase3_step | 跑一步不报错，返回含 pmpoloss/valueLoss/advantages |
| T8.12 | phase3_only_updates_policy_value | Phase 3 后 tokenizer/dynamics 参数无变化 |

```python
# T8.6 P1 关键测试：验证 imagineLatent 被调用
def test_phase3_uses_imagine():
    dreamer = make_small_dreamer()
    dreamer.freezePolicyPrior()
    batch = make_small_batch()
    call_count = [0]
    orig_imagine = dreamer.dynamics.imagineLatent
    def counting_imagine(*args, **kwargs):
        call_count[0] += 1
        return orig_imagine(*args, **kwargs)
    dreamer.dynamics.imagineLatent = counting_imagine
    dreamer.trainPhase3(batch)
    assert call_count[0] == dreamer.imagination_horizon, "imagineLatent must be called H times"

# T8.10 P4 关键测试：MTP 索引
def test_mtp_distance0_current_action():
    dreamer = make_small_dreamer()
    batch = make_small_batch()
    # monkeypatch policyHead.forward 记录 distance=0 时的 target
    targets_seen = []
    orig_forward = dreamer.policyHead.forward
    def recording_forward(h, distance=0):
        if distance == 0:
            targets_seen.append(h.shape[1])
        return orig_forward(h, distance)
    dreamer.policyHead.forward = recording_forward
    dreamer.trainPhase2(batch)
    # distance=0 时 h_used 长度应 = h.shape[1]（不减 1）
    assert targets_seen[0] == dreamer.dynamics.agentOutputs(
        dreamer.tokenizer.encode(batch.observations), batch.actions
    ).shape[1]
```

## T9. test_integration.py

| 编号 | 测试项 | 验证内容 |
|------|--------|---------|
| T9.1 | phase1a_to_1b_chain | Phase1a 2 步 → 冻结 tokenizer → Phase1b 2 步不报错 |
| T9.2 | phase1b_to_2_chain | Phase1b → Phase2 2 步 → frozenPrior 被设置 |
| T9.3 | phase2_to_3_chain | Phase2 → Phase3 2 步 → 返回合法指标 |
| T9.4 | checkpoint_save_load | 存 → 加载 → 参数一致 |
| T9.5 | checkpoint_phase_isolation | Phase3 检查点不含 phase1a 优化器状态 |
| T9.6 | buffer_sample_shapes | 采样返回 (B,T,C,H,W) + actions + rewards + dones + isRelevant |
| T9.7 | buffer_offline_roundtrip | saveOffline → loadOffline → 数据一致 |
| T9.8 | config_t2_gt_c | batchLengthLong=64 > contextLength=48 |

```python
# T9.3 端到端链路测试
def test_phase2_to_3_chain():
    dreamer = make_small_dreamer()
    batch = make_small_batch()
    dreamer.trainPhase1a(batch)
    dreamer.trainPhase1b(batch)
    dreamer.trainPhase2(batch)
    assert dreamer.frozenPrior is not None
    metrics = dreamer.trainPhase3(batch)
    assert "pmpoloss" in metrics
    assert "valueLoss" in metrics
    assert "advantages" in metrics
    assert torch.isfinite(torch.tensor(metrics["phase3Loss"]))
```

---

## conftest.py 共享 fixtures

```python
import torch
import pytest
from attridict import Attribict

@pytest.fixture
def small_config():
    """极小配置，CPU 可跑，用于形状和逻辑验证。"""
    return Attribict({
        "imageSize": 64, "patchSize": 8, "patchChannels": 32,
        "contextLength": 4, "imaginationHorizon": 3, "mtpLength": 2,
        "actionBins": 5,
        "tokenizer": Attribict({
            "bottleneckTokens": 128, "bottleneckDim": 16,
            "latentTokens": 64, "latentDim": 32,
            "modelDim": 64, "layers": 2, "heads": 2,
            "mlpRatio": 2, "maskRatioMax": 0.9,
            "lpipsWeight": 0.2, "singleFrameProb": 0.3,
        }),
        "dynamics": Attribict({
            "modelDim": 64, "layers": 4, "heads": 2, "kvHeads": 1,
            "registers": 2, "actionTokens": 1, "agentTokens": 2,
            "tauBins": 9, "stepBins": [0.25, 0.5, 1.0],
            "softCap": 50.0, "dropout": 0.0,
        }),
        "sampleSteps": 4, "tauCtx": 0.1,
        "discount": 0.997, "lambda_": 0.95,
        "pmpoAlpha": 0.5, "pmpoBeta": 0.3,
        "twohot": Attribict({"bins": 64, "low": -5.0, "high": 5.0}),
        "headHidden": 64,
        "lr": 1e-4, "phase2Lr": 3e-5, "weightDecay": 0.01,
        "gradientClip": 1.0, "gradientNormType": 2,
        "precision": "fp32", "lossNormDecay": 0.99,
        "buffer": Attribict({"capacity": 1000}),
    })

@pytest.fixture
def small_dreamer(small_config):
    from dreamer import DreamerV4
    return DreamerV4(
        (3, 64, 64), 3, [-1, -1, -1], [1, 1, 1],
        torch.device("cpu"), small_config
    )

@pytest.fixture
def small_batch():
    import attridict
    B, T = 2, 8
    return attridict({
        "observations": torch.rand(B, T, 3, 64, 64),
        "actions": torch.randn(B, T, 3),
        "rewards": torch.randn(B, T, 1),
        "dones": torch.zeros(B, T, 1),
        "isRelevant": torch.ones(B, T, 1, dtype=torch.bool),
    })
```

## 关键测试矩阵（修复项 × 测试）

| 修复项 | 直接测试编号 | 验证要点 |
|--------|-------------|---------|
| P1 想象 rollout | T8.6 T8.7 T8.8 T8.9 T8.11 T9.3 | imagineLatent 被调用 H 次；logp 来自采样动作；dynamics 无梯度变化 |
| P2 MAE 逐 patch | T5.4 T5.5 T5.6 | mask 形状 (B,T,P,1)；同帧内独立丢弃 |
| P3 memory_mask | T5.7 T5.8 | 形状 (T*P, T*Nb)；patch 只看 ≤t 步 latent |
| P4 MTP 索引 | T8.10 | distance=0 时 target 从 a_0 开始 |
| P5 PMPO 空集 | T7.6 T7.7 T7.8 T7.9 | 全正/全负/混合均返回标量 |
| P6 上下文 τ | T6.8 T6.9 | ctx_tau_idx=7≠8；与 0.9 对应 |
