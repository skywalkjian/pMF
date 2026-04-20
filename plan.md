# Kimi Residual Attention — 完善与验证计划

> 由 AI 研发团队 [Scientist / Engineer / Reviewer / DevOps] 于 2026-04-20 自主编写

---

## 一、现状分析

### 1.1 已有实现

`FullAttentionResidual` 类已在 `meanflow.py:210-229` 中实现。核心思想是用 depth-wise attention 替代加法残差连接：

- 维护所有历史层输出的 list
- 使用可学习 pseudo-query 向量对 history 做 softmax 加权聚合
- Query 零初始化 → 初始行为退化为均匀平均（安全的初始化策略）
- RMS Norm 仅用于 score 计算，加权求和使用原始 keys

### 1.2 发现的问题

| 编号 | 问题 | 严重程度 | 说明 |
|------|------|----------|------|
| B1 | 历史实验数据污染 | 高 | `naive_2gpu/20260420-005442` 实际使用了 `attn_impl=residual`，naive 基线被污染 |
| B2 | residual 2gpu 实验未做 FID 评估 | 中 | 有 5k checkpoint 但未运行 fid_eval.py |
| B3 | 5k steps 训练不够充分 | 中 | README 历史记录显示 10k steps 的 FID 更好 (37.52 vs 74.62) |

### 1.3 硬件

- 2x NVIDIA L20Z, 81GB VRAM 每卡
- PyTorch 2.5.1 + CUDA 11.8
- 对 CIFAR-10 小模型来说完全不用担心 OOM

---

## 二、代码重构方案

### 2.1 创建 `kimi_attention.py`

从 `meanflow.py` 提取以下组件到独立文件 `kimi_attention.py`：
- `rms_norm_last_dim()` 函数
- `FullAttentionResidual` 类

`meanflow.py` 改为 `from kimi_attention import FullAttentionResidual, rms_norm_last_dim`。

### 2.2 不做过度重构

现有实现在数学和工程上已经正确（经 [Reviewer] 审查通过），无需修改核心逻辑。重构仅限于文件组织。

---

## 三、实验方案

### 3.1 探针实验 (Probe Run)

```
torchrun --nproc_per_node=2 meanflow.py \
  --workdir ./runs/probe_residual_2gpu \
  --optimizer muon --attn-impl residual \
  --max-steps 200 --batch-size 128 --grad-accum-steps 1 \
  --eval-every 100 --sample-every 200 --save-every 200 \
  --sample-batch-size 8 --num-workers 4
```

**观察目标**：
- Loss 是否正常下降
- 每 step 耗时
- GPU 显存使用

### 3.2 正式训练

基于 probe 结果微调参数后，运行 10000 步正式训练（两组）：

**对照组 (naive)**:
```
torchrun --nproc_per_node=2 meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_naive_final \
  --optimizer muon --attn-impl naive \
  --max-steps 10000 --batch-size 128 --grad-accum-steps 1 \
  --eval-every 250 --sample-every 500 --save-every 500 \
  --sample-batch-size 8 --num-workers 4
```

**实验组 (residual)**:
```
torchrun --nproc_per_node=2 meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_residual_final \
  --optimizer muon --attn-impl residual \
  --max-steps 10000 --batch-size 128 --grad-accum-steps 1 \
  --eval-every 250 --sample-every 500 --save-every 500 \
  --sample-batch-size 8 --num-workers 4
```

### 3.3 FID 评估

每组训练完成后自动运行 50k 样本 FID 评估（双卡加速）：
```
torchrun --nproc_per_node=2 fid_eval.py \
  --run-dir <run_dir>/<timestamp> \
  --checkpoint best --data-root ./data \
  --num-samples 50000 --gen-bsz 128 --fid-batch-size 128 --isc
```

### 3.4 SOTA 目标

CIFAR-10 无条件/条件生成 FID 参考线：
- 当前项目最佳 (naive, 10k, 单卡): FID ≈ 37.52
- 目标: residual 的 FID < 37.52（优于 naive 基线）

---

## 四、自动化流水线

`auto_experiment.sh` 脚本包含：
1. 环境检查（GPU 可用性、数据集存在性）
2. naive 对照组训练 (10k steps)
3. residual 实验组训练 (10k steps)
4. 两组 FID 评估
5. 自动生成 `report.md` 汇总结果
6. 完善的错误捕获和日志记录

---

## 五、预期时间线

| 阶段 | 估计耗时 |
|------|----------|
| 代码重构 | 即时 |
| Probe run (200 steps) | ~3 分钟 |
| 正式训练 × 2 (10k steps each) | ~35 分钟/组 = ~70 分钟 |
| FID 评估 × 2 (50k samples) | ~15 分钟/组 = ~30 分钟 |
| **总计** | **~100 分钟** |

---

## 六、风险评估

| 风险 | 概率 | 对策 |
|------|------|------|
| Loss NaN | 低 | 脚本自动检测并以更小 LR 重试 |
| CUDA OOM | 极低 | L20Z 81GB 足够；若发生，降低 batch_size |
| FID 未优于 naive | 中 | 这是研究结果，如实记录 |
| DDP 通信异常 | 低 | NCCL 环境变量已在 meanflow.py 中配置 |
