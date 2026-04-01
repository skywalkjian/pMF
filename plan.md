# MNIST 主线、保留 CIFAR 扩展的 pMF 复现与面试规划

## 1. 这份规划解决什么问题

这份 `plan.md` 采用双轨方案：

1. 主线是 `MNIST POC`
2. 扩展线保留 `CIFAR10`

这样做的原因很现实：

- 时间紧时，`MNIST` 更容易形成完整闭环
- 面试更看重你有没有完成“问题定义 -> 实现 -> 实验 -> 分析 -> 总结”
- 已经做好的 `CIFAR10` 版本不丢，作为后续扩展成果保留

这份规划默认基于当前 `pMF` 子仓库状态执行，训练主入口是 [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py)，不是 [evaluate.py](/home/chen/githubku/pixelmeanflow/pMF/evaluate.py)。

---

## 2. 当前项目状态快照

### 2.1 当前训练入口

- 训练入口: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py)
- CLI 解析: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L904)
- 主训练循环: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L973)
- 模型工厂: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L729)
- 采样函数: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L673)

### 2.2 当前已经具备的能力

- `MNIST + U-Net + MeanFlow`
- `CIFAR10 + Transformer + MeanFlow`
- `CIFAR10 + Transformer + pMF`
- `CIFAR10 + Transformer + pMF + Muon`
- `CIFAR10 + Transformer + pMF + Muon + Residual Attention`

### 2.3 当前关键实现位置

- `U-Net baseline`: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L268)
- `Transformer backbone`: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L472)
- `MeanFlow loss`: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L552)
- `pMF loss`: [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py#L612)
- `Muon optimizer`: [muon.py](/home/chen/githubku/pixelmeanflow/pMF/optim/muon.py)

### 2.4 当前 git 演进链

已经完成的 5 个关键提交：

1. `57aca02` `refactor(meanflow): modularize MNIST baseline around meanflow.py`
2. `2ab65b7` `feat(meanflow): port baseline to CIFAR10 with transformer backbone`
3. `58a4759` `feat(pmf): add CIFAR10 pMF objective on shared transformer backbone`
4. `99ec9a9` `feat(optim): add Muon optimizer option for pMF training`
5. `cf446da` `feat(attn): implement residual attention for CIFAR10 pMF chain`

### 2.5 当前实验策略

从现在开始，实验策略改成：

- 主线优先做 `MNIST POC`
- 保留 `CIFAR10` 版本，不删除、不回退
- 面试时先讲 `MNIST` 的完整闭环
- 再把 `CIFAR10` 当作扩展能力和后续计划

---

## 3. 总体执行路线

整条路线分成两个轨道：

```text
主线:
MNIST 稳定训练
-> MNIST 上验证方法改动
-> 形成完整复现结论
-> 面试主故事

扩展线:
保留 CIFAR10 三组对比
-> 作为更复杂设定下的延展
-> 面试时作为加分项
```

优先级如下：

```text
必须完成:
MNIST 闭环
-> baseline 跑稳
-> pMF 跑通
-> 分析总结

有时间再做:
CIFAR10 三组比较
-> AdamW
-> Muon
-> Residual Attention
```

---

## 4. 阶段 A：训练前准备

### 4.1 目标

在正式跑长实验前，确认：

- 环境可训练
- 数据可下载
- 输出目录可写
- 命令参数不会配错
- 先用最小代价验证一条主线

### 4.2 必须知道的项目边界

- [evaluate.py](/home/chen/githubku/pixelmeanflow/pMF/evaluate.py) 主要还是 ImageNet 推理/FID 路径
- `MNIST` 和 `CIFAR10` 训练主逻辑都集中在 [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py)
- 目前最值得优先完成的是 `MNIST POC`
- `CIFAR10` 现在不是主战场，而是保留的扩展线

### 4.3 训练前检查清单

- `python -m py_compile meanflow.py optim/__init__.py optim/muon.py`
- 确认 `torch.cuda.is_available()` 为 `True`
- 确认数据目录可写: `pMF/data/`
- 确认输出目录可写: `pMF/runs/` 或自定义 `workdir`
- 确认 GPU 驱动、CUDA、PyTorch 版本正常

### 4.4 建议先做最短 smoke

优先先做 `MNIST baseline` 的设备级基准，记录：

- 每 step 用时
- 峰值显存
- loss 是否稳定
- sample 是否正常落盘

建议命令：

```bash
python pMF/meanflow.py \
  --dataset mnist \
  --backbone unet \
  --variant meanflow \
  --optimizer adam \
  --batch-size 128 \
  --max-steps 50 \
  --eval-every 25 \
  --sample-every 50 \
  --save-every 50 \
  --num-workers 4 \
  --workdir ./runs/bench_mnist_meanflow
```

### 4.5 这一阶段的产出

- 一条确认能稳定起跑的 `MNIST` 命令
- 一份大致 step time 估计
- 一个适合当前机器的 batch size

---

## 5. 阶段 B：MNIST 主线训练计划

### 5.1 为什么先做 MNIST

`MNIST POC` 最有价值的点不是“数据简单”，而是它更容易把研究流程做完整：

- baseline 明确
- 改动归因清楚
- 训练和样本观察都更快
- 更适合在面试里讲“我是怎么迭代方法的”

### 5.2 主线推荐目标

建议把 `MNIST` 主线拆成 3 步：

1. `MNIST + U-Net + MeanFlow`
2. `MNIST + Transformer + pMF`
3. `MNIST + Transformer + pMF + Muon`

如果时间还够，再看要不要加：

4. `MNIST + Transformer + pMF + Muon + Residual`

### 5.3 第一步：MNIST baseline

目标：

- 把最基础的 `MeanFlow` 路径跑稳
- 形成最早可展示结果

建议正式命令：

```bash
python pMF/meanflow.py \
  --dataset mnist \
  --backbone unet \
  --variant meanflow \
  --optimizer adam \
  --batch-size 256 \
  --max-steps 5000 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/mnist_meanflow_baseline
```

### 5.4 第二步：MNIST 上接 pMF

目标：

- 在 `MNIST` 上验证 `x-prediction -> u -> V -> v-loss`
- 形成“从 baseline 到方法改进”的故事

建议命令：

```bash
python pMF/meanflow.py \
  --dataset mnist \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --batch-size 256 \
  --max-steps 5000 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/mnist_pmf_adamw_naive
```

### 5.5 第三步：MNIST 上接 Muon

目标：

- 验证 `Muon` 是否能带来更快或更稳的早期下降

建议命令：

```bash
python pMF/meanflow.py \
  --dataset mnist \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl naive \
  --batch-size 256 \
  --max-steps 5000 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/mnist_pmf_muon_naive
```

### 5.6 第四步：如果还有时间再加 residual attention

目标：

- 验证 residual attention 是否在更小问题上也能带来可见差异

建议命令：

```bash
python pMF/meanflow.py \
  --dataset mnist \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl residual \
  --batch-size 256 \
  --max-steps 5000 \
  --eval-every 250 \
  --sample-every 500 \
  --save-every 500 \
  --num-workers 4 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/mnist_pmf_muon_residual
```

### 5.7 MNIST 主线最少要保存什么

每组至少保留：

- `config.json`
- `metrics.json`
- `checkpoints/last.pt`
- `samples/step_*.png`

### 5.8 MNIST 主线的检查点

每组实验在这些节点做人工检查：

1. `100 step`
2. `500 step`
3. `1000 step`
4. `2000 step`
5. `最终 step`

每次只看三件事：

1. loss 是否有限且稳定
2. sample 是否逐渐成形
3. checkpoint 是否正常保存

---

## 6. 阶段 C：保留 CIFAR10 扩展线

### 6.1 CIFAR10 的定位

`CIFAR10` 现在保留，但角色变了：

- 它不是当前最优先完成项
- 它是已经具备的扩展能力
- 它证明你不仅做过小数据 POC，还做过更复杂设定迁移

### 6.2 当前保留的 CIFAR10 三组主对比

保留这三组：

1. `pmf + adamw + naive`
2. `pmf + muon + naive`
3. `pmf + muon + residual`

### 6.3 CIFAR10 推荐正式配置

如果后面有时间再跑，推荐：

- `batch_size=512`
- `max_steps=10000`
- `eval_every=500`
- `sample_every=1000`
- `save_every=1000`
- `num_workers=8`
- `sample_batch_size=16`
- `seed=1024`

### 6.4 CIFAR10 三组命令

#### 组 1: `pmf + adamw + naive`

```bash
python pMF/meanflow.py \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer adamw \
  --attn-impl naive \
  --batch-size 512 \
  --max-steps 10000 \
  --eval-every 500 \
  --sample-every 1000 \
  --save-every 1000 \
  --num-workers 8 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/pmf_adamw_naive
```

#### 组 2: `pmf + muon + naive`

```bash
python pMF/meanflow.py \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl naive \
  --batch-size 512 \
  --max-steps 10000 \
  --eval-every 500 \
  --sample-every 1000 \
  --save-every 1000 \
  --num-workers 8 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/pmf_muon_naive
```

#### 组 3: `pmf + muon + residual`

```bash
python pMF/meanflow.py \
  --dataset cifar10 \
  --backbone transformer \
  --variant pmf \
  --optimizer muon \
  --attn-impl residual \
  --batch-size 512 \
  --max-steps 10000 \
  --eval-every 500 \
  --sample-every 1000 \
  --save-every 1000 \
  --num-workers 8 \
  --sample-batch-size 16 \
  --seed 1024 \
  --workdir ./runs/pmf_muon_residual
```

### 6.5 当前已有的 CIFAR10 smoke 结果

当前仓库里已经保留了三组 1-step smoke 产物，可作为“路径已跑通”的证据：

- [metrics.json](/home/chen/githubku/pixelmeanflow/pMF/tmp_stage5_pmf_adamw_naive/20260401-115727/metrics.json)
- [metrics.json](/home/chen/githubku/pixelmeanflow/pMF/tmp_stage5_pmf_muon_naive/20260401-115752/metrics.json)
- [metrics.json](/home/chen/githubku/pixelmeanflow/pMF/tmp_stage5_pmf_muon_residual/20260401-115813/metrics.json)

---

## 7. 阶段 D：结果整理

### 7.1 结果整理分两层

先整理 `MNIST` 主线，再补 `CIFAR10` 扩展线。

### 7.2 MNIST 至少要做的一张表

建议表头：

| Experiment | Final train loss | Best val loss | Step time | Notes |
| --- | --- | --- | --- | --- |
| `mnist meanflow baseline` |  |  |  |  |
| `mnist pmf + adamw` |  |  |  |  |
| `mnist pmf + muon` |  |  |  |  |
| `mnist pmf + muon + residual` |  |  |  |  |

### 7.3 CIFAR10 扩展表

建议表头：

| Experiment | Final train loss | Best val loss | Step time | Notes |
| --- | --- | --- | --- | --- |
| `pmf + adamw + naive` |  |  |  |  |
| `pmf + muon + naive` |  |  |  |  |
| `pmf + muon + residual` |  |  |  |  |

### 7.4 样本图对比建议

`MNIST` 推荐看：

- `step 500`
- `step 1000`
- `step 5000`

`CIFAR10` 推荐看：

- `step 1000`
- `step 5000`
- `step 10000`

### 7.5 结果分析时优先回答的问题

主线 `MNIST`：

1. pMF 相比 baseline 是否更易学
2. Muon 是否更快进入下降区间
3. residual attention 是否有额外帮助

扩展线 `CIFAR10`：

1. 三组主线是否都稳定可训练
2. Muon 与 AdamW 的早期趋势是否一致
3. residual attention 是否改变早期样本结构

### 7.6 不要过度解释的地方

如果 `MNIST` 跑得更完整，而 `CIFAR10` 只跑了较短步数，就不要说：

- “我完整复现了论文”
- “我证明了某个组件一定最强”

更稳妥的说法是：

- “我在 `MNIST` 上完成了完整 POC 验证”
- “我把同一条方法链迁移到 `CIFAR10` 做了扩展实验”

---

## 8. 阶段 E：复现项目分析总结

### 8.1 分析应该从哪几个层面讲

建议按这 4 层来写：

1. 论文思想
2. 工程实现
3. 实验现象
4. 局限与下一步

### 8.2 论文思想层

你至少要能清楚讲出：

- MeanFlow 学的是 `u(z_t, r, t)` 这类 two-time quantity
- pMF 不直接预测 `u`
- pMF 用 `x-prediction -> u -> V -> v-loss`
- 这样做的动机是让输出空间更接近图像流形

### 8.3 工程实现层

你要能对着代码解释：

- 为什么训练统一收敛到 [meanflow.py](/home/chen/githubku/pixelmeanflow/pMF/meanflow.py)
- `MeanFlowLoss` 和 `PixelMeanFlowLoss` 的区别
- `generate_meanflow()` 和 `generate_pmf()` 的区别
- `Muon` 为什么只给 transformer block 内矩阵权重
- `Residual attention` 为什么传的是 `prev_logits`

### 8.4 实验现象层

建议先围绕 `MNIST` 写出明确结论模板：

```text
在 MNIST 小规模 POC 设置下，
pMF 相比 MeanFlow baseline 更容易/更难训练；
Muon 相比 AdamW 在早期 loss 下降速度上更快/相近/更不稳定；
Residual attention 带来/没有带来明显额外收益。
```

然后再写 `CIFAR10` 扩展结论：

```text
在 CIFAR10 小 transformer 设置下，
三组主线都能稳定起跑；
Muon 和 residual attention 在早期阶段表现出某些趋势，
但仍需要更长训练和更正式指标验证。
```

### 8.5 局限性层

必须明确指出：

- 当前主闭环是 `MNIST POC`
- `CIFAR10` 是扩展线，不是论文规模复现
- 没有做正式 FID
- backbone 不是官方 ImageNet 大模型
- 单 seed 结论可能不稳定

---

## 9. 阶段 F：用于面试的项目表达

### 9.1 一句话介绍

推荐版本：

> 我做了一个从 MeanFlow 到 pixel MeanFlow 的研究型复现，先在 `MNIST` 上构建了完整 POC，验证了 `pMF`、`Muon` 和 `Residual Attention` 对 one-step generation 训练动态的影响，然后把同一条实验链迁移到 `CIFAR10` 作为扩展验证。

### 9.2 3 分钟讲法

可以按下面顺序讲：

1. 论文问题
   - one-step + pixel-space 很难学
2. 核心方法
   - pMF 用 `x-prediction` 替代直接 `u-prediction`
3. 我的策略
   - 先做 `MNIST POC`
   - 再保留并扩展到 `CIFAR10`
4. 我的工作
   - 整理训练入口
   - 实现 pMF、Muon、Residual Attention
5. 我的发现
   - 哪些改动有效
   - 哪些结论还需要更大规模验证

### 9.3 高频问题准备

#### Q1: 你到底复现了什么？

回答方向：

- 不是完整 ImageNet 论文规模复现
- 是一条结构完整的研究型复现链
- 主线在 `MNIST` 上闭环，扩展到 `CIFAR10`

#### Q2: 为什么先做 MNIST？

回答方向：

- 为了更快完成方法验证
- 为了让每一步改动都更容易归因
- 为了先做完一个完整闭环，再迁移复杂设定

#### Q3: 为什么还保留 CIFAR10？

回答方向：

- 因为它证明方法不是只能在 toy setting 起作用
- 也能体现我有把小规模验证迁移到更复杂设定的能力

#### Q4: 你最大的工程贡献是什么？

回答方向：

- 把原本分散的逻辑统一到一个训练中心
- 让 baseline、pMF、Muon、Residual Attention 能在同一实验框架里可比较

#### Q5: 你怎么看自己的局限？

回答方向：

- 目前最扎实的是 `MNIST POC`
- `CIFAR10` 还不是论文规模验证
- 需要更长训练、更多 seed 和正式指标

### 9.4 最容易踩坑的表达

不要说：

- “我完整复现了论文”
- “我证明了 Muon 一定更优”
- “Residual Attention 一定提升最终质量”

更好的说法是：

- “我完成了一个完整的 POC 复现，并做了更复杂设定迁移”
- “我观察到了早期训练现象，但还需要更正式验证”

---

## 10. 建议执行日程

### 方案 A：今天就要有完整故事

#### 上午

- 跑 `MNIST baseline`
- 跑 `MNIST pMF`

#### 下午

- 跑 `MNIST pMF + Muon`
- 如果还能跑，再加 `Residual Attention`

#### 晚上

- 整理结果
- 写复现结论
- 准备面试稿

### 方案 B：2 到 3 天稳妥版

#### 第 1 天

- 跑通全部 `MNIST` 主线
- 固定命令和日志目录

#### 第 2 天

- 整理 `MNIST` 对比结果
- 补 `CIFAR10` 扩展实验

#### 第 3 天

- 汇总分析
- 准备面试表达

---

## 11. 最终交付物清单

### 11.1 实验产物

- `MNIST` 主线各组 `workdir`
- `CIFAR10` 扩展线 `workdir`
- 每组 `config.json`
- 每组 `metrics.json`
- 每组 sample 图
- 每组最后 checkpoint

### 11.2 分析产物

- 一张 `MNIST` 主结果表
- 一张 `CIFAR10` 扩展结果表
- 一页复现分析总结
- 一页方法理解总结

### 11.3 面试产物

- 30 秒介绍版本
- 3 分钟讲解版本
- 高频问答提纲

---

## 12. 最后提醒

这次项目最有价值的地方，不只是“把代码跑起来”，而是你已经形成了一条很像研究工作的路径：

```text
先在 MNIST 做可控 POC
-> 在小系统里验证 pMF 思路
-> 再引入 Muon 和 Residual Attention
-> 保留 CIFAR10 作为迁移扩展
-> 形成复现分析
-> 最后沉淀成面试表达
```

主线讲 `MNIST`，扩展讲 `CIFAR10`，这是现在最稳、也最适合你时间状态的方案。

