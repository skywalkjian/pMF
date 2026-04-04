# 实验运行指南

进入目录：

```bash
cd pMF
pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

训练入口：

```bash
python meanflow.py ...
```

FID 入口：

```bash
python fid_eval.py ...
```

每次训练会在 `--workdir/<timestamp>/` 下生成：

- `config.json`
- `metrics.json`
- `samples/`
- `checkpoints/`

## 1. 先跑 smoke

```bash
python meanflow.py \
  --workdir ./runs/smoke_pmf_cifar10 \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --max-steps 20 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --eval-every 20 \
  --sample-every 20 \
  --save-every 20 \
  --sample-batch-size 4 \
  --num-workers 0
```

## 2. 三组 CIFAR10 主实验

统一原则：

- 三组实验保持同一套 micro-batch、`grad_accum_steps`、transformer 超参
- 当前默认先用 `batch_size=32 + grad_accum_steps=8`
- 这相当于 effective batch 约 `256`
- 如果仍然 OOM，再把三组一起降到 `batch_size=16 + grad_accum_steps=16`

### pMF + AdamW + naive

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_adamw_transformer_naive \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --max-steps 5000 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --sample-batch-size 8 \
  --num-workers 4
```

### pMF + Muon + naive

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_naive \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl naive \
  --max-steps 5000 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --sample-batch-size 8 \
  --num-workers 4
```

### pMF + Muon + residual

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_residual \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl residual \
  --max-steps 5000 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --sample-batch-size 8 \
  --num-workers 4
```

## 3. 恢复训练

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_residual \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl residual \
  --max-steps 10000 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --sample-batch-size 8 \
  --num-workers 4 \
  --resume ./runs/cifar10_pmf_muon_transformer_residual/<timestamp>/checkpoints/last.pt
```

## 4. 跑 FID

默认用 `best.pt`，参考集用 `cifar10-train`。

### 单组命令模板

```bash
python fid_eval.py \
  --run-dir ./runs/cifar10_pmf_adamw_transformer_naive/<timestamp> \
  --checkpoint best \
  --data-root ./data \
  --num-samples 50000 \
  --gen-bsz 128 \
  --fid-batch-size 128 \
  --device cuda
```

### 三组命令

```bash
python fid_eval.py --run-dir ./runs/cifar10_pmf_adamw_transformer_naive/<timestamp> --checkpoint best --data-root ./data --num-samples 50000 --gen-bsz 128 --fid-batch-size 128 --device cuda
python fid_eval.py --run-dir ./runs/cifar10_pmf_muon_transformer_naive/<timestamp> --checkpoint best --data-root ./data --num-samples 50000 --gen-bsz 128 --fid-batch-size 128 --device cuda
python fid_eval.py --run-dir ./runs/cifar10_pmf_muon_transformer_residual/<timestamp> --checkpoint best --data-root ./data --num-samples 50000 --gen-bsz 128 --fid-batch-size 128 --device cuda
```

结果会写到：

- `fid/metrics_best_50000.json`

如果你想保留生成图：

```bash
python fid_eval.py \
  --run-dir ./runs/cifar10_pmf_muon_transformer_residual/<timestamp> \
  --checkpoint best \
  --data-root ./data \
  --num-samples 50000 \
  --gen-bsz 128 \
  --fid-batch-size 128 \
  --device cuda \
  --keep-samples
```

## 5. 结果怎么看

训练时主要看：

- `metrics.json`
- `samples/step_0000500.png`
- `samples/step_0002000.png`
- `samples/step_0005000.png`
- `checkpoints/best.pt`

FID 主要看：

- `fid/metrics_best_50000.json`
- 其中的 `metrics.frechet_inception_distance`

## 6. 重要说明

- 修复后的 `pMF` 只支持 `transformer`。
- `Muon` 只用于 `variant=pmf`。
- 如果训练仍然 OOM，三组一起降到：
  `--batch-size 16 --grad-accum-steps 16 --sample-batch-size 4`
- 如果还是 OOM，再把三组一起降到：
  `--hidden-size 192 --depth 6`
- 你 2026-04-01 那三组旧 CIFAR10 run 的 checkpoint 目前读不出来，不能直接补算 FID；需要用当前代码重新跑并保存新 checkpoint。
