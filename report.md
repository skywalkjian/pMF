# Kimi Residual Attention — Experiment Report

> 2026-04-20 实验完成，手动修正结果表格

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | PmfTransformer (hidden=256, depth=8, heads=4) |
| Dataset | CIFAR-10 (32x32) |
| Optimizer | Muon + AdamW hybrid |
| Effective Batch Size | 128 x 2 GPUs = 256 |
| Training Steps | 10,000 |
| GPUs | 2x NVIDIA L20Z (81GB) |
| FID Samples | 50,000 |
| Sampling | 1-step, omega=1.0, t_min=0.0, t_max=1.0 |

## Results

| Experiment | Steps | Train Loss | Best Val Loss | Best Step | FID (50k) | IS Mean |
|------------|-------|-----------|---------------|-----------|-----------|---------|
| Muon + naive | 10,000 | 1.856936 | 1.861646 | 4,500 | 74.62 | 4.72 |
| Muon + residual | 10,000 | 1.855834 | 1.859786 | 9,500 | **57.42** | **5.91** |

## Key Findings

1. **FID: residual 大幅优于 naive**，从 74.62 降至 57.42（改善 23%）
2. **IS: residual 明显优于 naive**，从 4.72 提升至 5.91（改善 25%）
3. **Val loss 差距微小**（1.8616 vs 1.8598），但 FID/IS 差距显著
4. **Residual 最佳 checkpoint 在 step 9,500**，而 naive 在 step 4,500 就停止改善

## Run Directories

- **Naive**: `runs/cifar10_pmf_muon_transformer_naive_final/20260420-033936`
- **Residual**: `runs/cifar10_pmf_muon_transformer_residual_final/20260420-040008`

## Logs

- Main: `agent_team_run.log`
- Naive training: `experiment_logs/train_naive_20260420-033921.log`
- Residual training: `experiment_logs/train_residual_20260420-033921.log`
- Naive FID: `experiment_logs/fid_naive_20260420-033921.log`
- Residual FID: `experiment_logs/fid_residual_20260420-033921.log`
