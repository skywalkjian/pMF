# 实验运行指南

训练入口只有一个：

```bash
python meanflow.py ...
```

先进入项目目录并安装依赖：

```bash
cd pMF
pip install -r requirements.txt
```

数据会自动下载到 `./data`。每次运行都会在 `--workdir` 下生成一个时间戳目录，里面有：

- `config.json`
- `metrics.json`
- `samples/`
- `checkpoints/`

## 1. 先跑 smoke

先确认训练、采样、checkpoint 都正常。

### MeanFlow baseline smoke

```bash
python meanflow.py \
  --workdir ./runs/smoke_meanflow_mnist \
  --dataset mnist \
  --backbone unet \
  --variant meanflow \
  --optimizer adam \
  --max-steps 20 \
  --batch-size 64 \
  --eval-every 20 \
  --sample-every 20 \
  --save-every 20 \
  --num-workers 0
```

### pMF smoke

```bash
python meanflow.py \
  --workdir ./runs/smoke_pmf_mnist \
  --dataset mnist \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --max-steps 20 \
  --batch-size 64 \
  --eval-every 20 \
  --sample-every 20 \
  --save-every 20 \
  --num-workers 0
```

## 2. 正式实验

## 2.1 MeanFlow baseline

这是旧 baseline，对照用。

```bash
python meanflow.py \
  --workdir ./runs/mnist_meanflow_baseline \
  --dataset mnist \
  --backbone unet \
  --variant meanflow \
  --optimizer adamw \
  --max-steps 5000 \
  --batch-size 256 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4
```

## 2.2 MNIST 上的 pMF

这是修复后的 pMF 验证实验。

```bash
python meanflow.py \
  --workdir ./runs/mnist_pmf_transformer_naive \
  --dataset mnist \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --max-steps 5000 \
  --batch-size 256 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4
```

## 2.3 CIFAR10 主实验

这是当前最重要的三组。

### pMF + AdamW + naive

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_adamw_transformer_naive_v2 \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --max-steps 5000 \
  --batch-size 256 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4
```

### pMF + Muon + naive

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_naive_v2 \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl naive \
  --max-steps 5000 \
  --batch-size 256 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4
```

### pMF + Muon + residual

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_residual_v2 \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl residual \
  --max-steps 5000 \
  --batch-size 256 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4
```

## 3. 恢复训练

`--resume` 要指向具体 checkpoint 文件，不是目录。

```bash
python meanflow.py \
  --workdir ./runs/cifar10_pmf_muon_transformer_residual_v2 \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl residual \
  --max-steps 10000 \
  --batch-size 256 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4 \
  --resume ./runs/cifar10_pmf_muon_transformer_residual_v2/<timestamp>/checkpoints/last.pt
```

## 4. 你主要看什么

- `metrics.json`
- `samples/step_0000500.png`
- `samples/step_0002000.png`
- `samples/step_0005000.png`
- `checkpoints/best.pt`

建议优先看 `best.pt` 对应阶段，不要只看最后一步。

## 5. 重要说明

- 修复后的 `pMF` 只支持 `transformer`，不再支持 `U-Net + pMF`。
- 你之前旧版 `pMF` 的结果和 checkpoint 都不该再继续比较，建议全部按新实验重跑。
- `Muon` 只用于 `variant=pmf`。
- 如果显存不够，先把 `--batch-size` 从 `256` 降到 `128` 或 `64`。
- 如果只是想快速看趋势，先跑 `1000` step，不要一上来就长跑。
