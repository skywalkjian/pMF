# 从 MeanFlow 到 pMF：一个训练链补齐、实验归因与负结果分析项目

## 30 秒版摘要

这个项目的核心不是“把一篇论文跑出来”，而是把一个原本以推理为主的 PyTorch 仓库，补成一条可训练、可比较、可分析的研究型实验链。  
我把训练统一收束到 [meanflow.py](./meanflow.py)，构造了从 `MNIST + U-Net + MeanFlow` 到 `CIFAR10 + Transformer + pMF`、再到 `Muon` 和 `Residual Attention` 的完整演进路径，并用真实实验结果分析方法什么时候有效、什么时候失效。

当前最可信的结论是：

- `CIFAR10` 上三条主比较路径都已跑通到 `5000 step`，足以观察早期结构趋势。
- `MNIST` 上的负结果非常关键：`Transformer + pMF` 明显不如 baseline，而 `pMF + U-Net` 会出现“中期像、后期糊”。
- 这个项目最能体现我能力的部分，不是调参，而是构造出一条能从负结果中得到研究判断的实验链。

## 证据等级说明

为了避免把项目写成印象流，下面的结论统一按三类证据表述：

- `[代码事实]`：可以直接在当前仓库代码中定位。
- `[实验观察]`：来自现有 `runs/`、`metrics.json` 和 sample 图。
- `[当前判断]`：基于现有代码和实验现象做出的研究判断，不写成严格定论。

## 1. 项目背景与定位

`pMF` 关注的是一个很难的问题：

- `one-step` 生成
- `latent-free`
- 直接在 `pixel-space` 上建模

这意味着模型既不能依赖 latent tokenizer，也不能靠很多步数值积分慢慢逼近，而要在极少步甚至一步内直接从噪声走到图像。

我面对的现实起点是：

- 上游 PyTorch 仓库主要提供推理与预训练权重。
- 当前目录里真实有用来训练和实验比较的主入口，是我补齐后的 [meanflow.py](./meanflow.py)。

因此，我把这个项目的目标定义成：

1. 补齐当前 PyTorch 仓库里的训练链。
2. 构造一条可比较的实验演进路径。
3. 用 `MNIST` 做快速验证与失败分析，用 `CIFAR10` 做更有代表性的主线比较。
4. 不回避负结果，而是通过负结果反过来理解方法边界。

这份说明只围绕当前仓库和当前改动展开，不依赖其他外部代码库叙事。

## 2. 项目目标

### 2.1 工程目标

- 把训练逻辑统一到 [meanflow.py](./meanflow.py)。
- 让 dataset、backbone、variant、optimizer、sample、checkpoint 在同一个入口下编排。
- 保留上游 [evaluate.py](./evaluate.py) 和 `models/` 中的推理逻辑，不和训练链混在一起。

### 2.2 实验目标

- 建立 `MNIST + U-Net + MeanFlow` baseline。
- 在 `CIFAR10 + Transformer` 骨架上加入 `pMF`。
- 在统一骨架上比较 `pMF / pMF + Muon / pMF + Muon + Residual Attention`。

### 2.3 研究目标

- 理解 `pMF` 相比 `MeanFlow` 改变的究竟是什么。
- 观察优化器和 attention 机制是否改变早期结构学习。
- 通过负结果判断哪些变化真的有效，哪些变化只是“改了但没帮上忙”。

## 3. 训练链与代码结构

### 3.1 训练主中心

`[代码事实]` 当前训练主中心在 [meanflow.py](./meanflow.py)，它统一承担了：

- `parse_args()`
- `build_datasets()`
- `build_model()`
- `build_optimizer()`
- `build_loss()`
- `train(config)`
- `generate_samples(...)`

也就是说，这个项目不是把实验分散在很多零碎脚本里，而是有一个明确的训练编排中心。

### 3.2 当前仓库中与本项目最相关的文件

```text
pMF/
├── meanflow.py
├── optim/
│   └── muon.py
├── evaluate.py
├── models/
├── runs/
└── assets/showcase/
```

- [meanflow.py](./meanflow.py)：训练主线
- [optim/muon.py](./optim/muon.py)：Muon + AdamW 的混合优化器实现
- [evaluate.py](./evaluate.py)：上游推理/评估逻辑
- `runs/`：实验产物
- `assets/showcase/`：我为 README 和展示文档固定出来的代表性图片

### 3.3 项目 Pipeline

```text
dataset
  -> corruption / time sampling
  -> backbone
  -> objective
  -> optimizer
  -> train loop
  -> sample / checkpoint / metrics
```

如果聚焦到当前 `pMF + Transformer` 主线，可以进一步写成：

```text
image x
  -> sample r, t, epsilon
  -> build z_t
  -> model predicts x_pred
  -> analytic transform to u
  -> JVP gives d u / d t
  -> build V
  -> optimize v-loss
```

## 4. 工程演进链

这个项目不是“一步改成最终形态”，而是通过 5 个提交逐步搭起来的：

| Commit | 阶段 | 作用 |
| --- | --- | --- |
| `57aca02` | Commit 1 | 模块化 `MNIST + U-Net + MeanFlow` baseline |
| `2ab65b7` | Commit 2 | 迁移到 `CIFAR10 + Transformer + MeanFlow` |
| `58a4759` | Commit 3 | 增加 `pMF` 训练目标 |
| `99ec9a9` | Commit 4 | 增加 `Muon` 优化器选项 |
| `cf446da` | Commit 5 | 增加当前版本的 `Residual Attention` |

这条链的价值在于：每一步都有明确语义，可以分别追踪“换了什么”和“为什么换”。

## 5. 方法理解

### 5.1 MeanFlow 在这里学的是什么

`[代码事实]` 当前 baseline 的核心目标在 [meanflow.py](./meanflow.py) 的 `MeanFlowLoss`。

它不是直接对网络做一个普通的图像重建，而是通过 two-time 量和 `torch.func.jvp` 去构造平均速度相关的监督。直觉上可以理解成：

- `z_t` 是真实图像和噪声之间的中间点
- 网络在学习如何从这条连续轨迹上恢复“往图像方向走”的信息

### 5.2 pMF 改了什么

`[代码事实]` 当前 `pMF` 路径在 [meanflow.py](./meanflow.py) 的 `PixelMeanFlowLoss`。

在这条路径里：

```text
x_pred -> u -> V -> v-loss
```

也就是：

1. 网络先输出更像图像的 `x_pred`
2. 再解析变换到 `u`
3. 再借助 `JVP` 和 `du/dt` 构造 `V`
4. 最后仍然在速度空间做 `v-loss`

`[当前判断]` 这让我对 `pMF` 的理解不是“简单换个 loss”，而是：

> 它在问，pixel-space 网络更容易学的输出形式究竟是什么。

### 5.3 Muon 在当前项目中做了什么

`[代码事实]` 当前参数分组逻辑在 [meanflow.py](./meanflow.py) 的 `build_optimizer()`，具体优化器实现在 [optim/muon.py](./optim/muon.py)。

现在这版实现不是“所有参数都走 Muon”，而是：

- transformer block 里的矩阵权重走 `Muon`
- patch embedding、norm、bias、其他参数走 AdamW 风格更新

`[当前判断]` 这样的好处是：

- 保留 Muon 在矩阵权重上的特点
- 避免把所有非矩阵参数硬塞进 Muon 带来的不稳定
- 更适合研究型对比，而不是激进重写

### 5.4 当前 Residual Attention 的实现逻辑

`[代码事实]` attention 相关逻辑在 [meanflow.py](./meanflow.py) 的 `TransformerAttention` 和 `TransformerBlock`。

当前实现的思路很直接：

```text
current_logits = qk_logits
if prev_logits exists:
    current_logits = current_logits + prev_logits
attn = softmax(current_logits)
```

也就是把上一层的 attention logits 继续传给下一层，让跨层 attention pattern 保持连续性。

`[当前判断]` 当前这个版本更适合作为“跨层 attention 累加”的工程实验，而不是把它当成一套独立的大型新机制。

## 6. 实验设置

### 6.1 CIFAR10 主对比

`[代码事实]` 当前三个主 run 的 config 完全可追溯，并且主比较只改两类变量：

- `optimizer`
- `attn_impl`

共同设置如下：

- `dataset = cifar10`
- `backbone = transformer`
- `variant = pmf`
- `seed = 1024`
- `batch_size = 256`
- `max_steps = 5000`
- `patch_size = 4`
- `hidden_size = 256`
- `depth = 8`
- `num_heads = 4`
- `mlp_ratio = 4.0`

三组主比较分别是：

1. `pMF + AdamW + naive`
2. `pMF + Muon + naive`
3. `pMF + Muon + residual`

### 6.2 MNIST 辅助观察

`MNIST` 这边我主要保留了三条结果：

1. `MeanFlow baseline + U-Net`
2. `pMF + Transformer`
3. `pMF + U-Net`

这三条路径帮助我回答的不是“谁分数更高”，而是：

- baseline 为什么在低维任务里很强
- `pMF` 在低维任务里为什么不一定占优
- 视觉质量为什么可能和 loss 不同步

## 7. 实验结果：CIFAR10 主线

### 7.1 Step 2000：看早期结构

| `pMF + AdamW + naive` | `pMF + Muon + naive` | `pMF + Muon + residual` |
| --- | --- | --- |
| ![cifar adamw naive step2000](./assets/showcase/cifar_adamw_naive_step2000.png) | ![cifar muon naive step2000](./assets/showcase/cifar_muon_naive_step2000.png) | ![cifar muon residual step2000](./assets/showcase/cifar_muon_residual_step2000.png) |

`[实验观察]` 在当前固定 seed 的 sample 图里，`Muon + residual` 在 `step 2000` 已经出现更明显的区域结构和颜色组织。

### 7.2 Step 5000：看阶段性结果

| `pMF + AdamW + naive` | `pMF + Muon + naive` | `pMF + Muon + residual` |
| --- | --- | --- |
| ![cifar adamw naive step5000](./assets/showcase/cifar_adamw_naive_step5000.png) | ![cifar muon naive step5000](./assets/showcase/cifar_muon_naive_step5000.png) | ![cifar muon residual step5000](./assets/showcase/cifar_muon_residual_step5000.png) |

`[实验观察]` 到 `step 5000` 时，三组都已经从纯噪声走向“有颜色块和局部结构的图像”，但还没有达到可以讨论最终图像质量的阶段。

### 7.3 当前能安全引用的数值

`[代码事实] + [实验观察]`

| 实验 | `best_val_loss` | `last train loss` |
| --- | ---: | ---: |
| `pMF + AdamW + naive` | `0.2431` | `0.2822` |
| `pMF + Muon + naive` | `0.2492` | `0.2481` |
| `pMF + Muon + residual` | `0.2378` | `0.2438` |

`[当前判断]` 我对这组三组结果的最克制表述是：

- `Muon + residual` 当前 run 的 `best_val_loss` 最低。
- `Muon + naive` 在尾段 loss 上更平稳。
- `AdamW + naive` 的数值也可以回到较低区间，但中间训练波动更明显。

### 7.4 我如何解读这组结果

`[当前判断]`

- `5000 step` 足以判断“是否开始学到结构”，不足以判断“最终谁最好”。
- 这组结果更适合展示为早期趋势比较，而不是最终性能排行榜。
- 如果要把结论做实，下一步必须加更长训练、更多 seed 和正式指标。

## 8. 负结果与研究判断：MNIST

### 8.1 `Transformer + pMF` 不如 baseline

| `MeanFlow baseline + U-Net @ step 500` | `pMF + Transformer @ step 500` |
| --- | --- |
| ![mnist baseline step500](./assets/showcase/mnist_meanflow_baseline_step500.png) | ![mnist transformer step500](./assets/showcase/mnist_pmf_transformer_step500.png) |

`[实验观察]` baseline 在很早的阶段就出现了清晰数字结构，而 `Transformer + pMF` 仍然接近破碎噪声。

`[当前判断]` 这说明至少在 `MNIST` 这种低维简单问题上：

- 卷积归纳偏置很强
- 同时替换 backbone 和 objective 会导致归因困难
- `pMF` 不会因为目标更复杂就自然更优

### 8.2 `pMF + U-Net` 的“中期像、后期糊”

| `pMF + U-Net @ step 500` | `pMF + U-Net @ step 5000` |
| --- | --- |
| ![mnist unet step500](./assets/showcase/mnist_pmf_unet_step500.png) | ![mnist unet step5000](./assets/showcase/mnist_pmf_unet_step5000.png) |

`[实验观察]` 中期样本已经能看出比较接近真实数字的轮廓，但后期图像反而变得更模糊、更平均。

`[当前判断]` 这提示了两个重要事实：

1. loss 下降并不保证视觉质量单调提升。
2. 在当前没有 EMA 的实现里，中间 checkpoint 很可能比最后 checkpoint 更适合展示。

### 8.3 我从 MNIST 负结果中得到的判断

`[当前判断]`

1. `pMF` 的优势更可能出现在更高维、更复杂的 pixel-space 问题里，而不是 `MNIST` 这种简单设定。
2. backbone 和 objective 必须拆开比较，否则无法知道真正有效的变化是什么。
3. 研究工作不能只展示正结果，负结果同样是形成判断的证据。

## 9. 这个项目体现的工程与研究能力

我认为这个项目能体现出的能力，不是单点调参，而是下面四个方面。

### 9.1 工程结构化能力

- 把训练从零散脚本整理成单一入口
- 把 dataset/backbone/variant/optimizer 统一到一个可配置系统里
- 把 sample、checkpoint、metrics 统一输出到可追溯目录

### 9.2 实验设计能力

- 把实验组织成一条明确的演进链
- 在 CIFAR10 主线中保持大部分变量不变，只比较关键改动
- 用 `MNIST` 作为快速验证和失败分析来源

### 9.3 结果归因能力

- 不把早期结果夸成最终结论
- 不因为有负结果就回避展示
- 能把样本现象、loss 曲线趋势和方法结构联系起来解释

### 9.4 研究判断能力

- 能从“为什么没成功”中形成下一步研究方向
- 能明确区分代码事实、实验观察和当前判断
- 能知道哪些结论现在可以说，哪些还不能说

## 10. 当前局限与下一步

### 10.1 当前局限

- 当前不是论文原始 ImageNet 规模复现。
- 当前 `CIFAR10` 只跑到 `5000 step`，更适合看早期趋势。
- 当前还没有正式 FID/KID 或更多 seed 的系统比较。
- 当前 README 中只放了 sample 图，没有现成 TensorBoard 曲线图。

### 10.2 下一步

我认为接下来最值得补的工作是：

1. 把 `CIFAR10` 主对比扩展到 `10000-20000 step`
2. 增加更多 seed，验证当前趋势是否稳定
3. 补 TensorBoard loss 曲线和固定 seed 的时间线图
4. 加更严格的对照，比如 `Transformer + MeanFlow` 对比 `Transformer + pMF`
5. 若要继续做 attention 研究，再单独设计更系统的实现与比较

## 11. 面试附录

### 11.1 1 分钟讲稿

我做了一个围绕 `MeanFlow` 和 `pMF` 的研究型复现项目。原始 PyTorch 仓库主要提供推理代码，所以我先把训练统一补到 `meanflow.py`，然后构造了一条从 `MNIST + U-Net + MeanFlow baseline` 到 `CIFAR10 + Transformer + pMF`、再到 `Muon` 和 `Residual Attention` 的实验链。

这个项目最有价值的地方不是跑出一张最好看的图，而是我把实验组织成了可比较的链条，并且把负结果也纳入分析。比如 `MNIST + Transformer + pMF` 明显不如 baseline，而 `pMF + U-Net` 会出现中期像、后期糊的现象。这让我判断 `pMF` 更可能适用于高维 pixel-space，而不是低维简单数据。

所以这个项目体现的不是单点调参，而是方法理解、工程实现、实验归因和研究判断。

### 11.2 3 分钟讲稿

这个项目的起点是：`pMF` 论文关注 one-step、latent-free、pixel-space 图像生成，但当前 PyTorch 仓库主要是推理代码，缺少完整训练链。所以我做的第一步不是直接跑结果，而是把训练统一整理到 `meanflow.py`，让它负责数据、模型、目标函数、优化器、采样和保存。

在这条训练链上，我分 5 步构造了一个可比较的演进路径：先整理 `MNIST + U-Net + MeanFlow` baseline，然后迁移到 `CIFAR10 + Transformer + MeanFlow`，再增加 `pMF` 的 `x-prediction -> u -> V -> v-loss` 训练路径，然后加入 `Muon`，最后加入当前版本的 `Residual Attention`。这样每一步改动都在同一框架里可比较，而不是换一整套实现。

实验上，我最后把 `CIFAR10` 作为主线，因为它更接近高维 pixel-space 问题。当前三组 `CIFAR10` 主对比都跑到了 `5000 step`，可以看到 `Muon + residual` 更早出现结构，而且当前 run 下的 `best_val_loss` 也最低。但我不会把这说成最终结论，因为 `5000 step` 更适合看早期趋势，不足以比较最终质量。

另一方面，`MNIST` 上的负结果反而非常有价值。`Transformer + pMF` 明显不如 baseline，而 `pMF + U-Net` 还出现了中期像、后期糊。这说明方法不会在所有问题上都自动更优，也说明生成模型里 loss 和视觉质量不一定同步。

所以我觉得这个项目最像研究工作的地方是：我不是单纯把代码跑通，而是通过正反结果一起形成方法边界判断。

### 11.3 高频追问与回答

#### Q1. 你到底复现了什么？

我没有把它说成完整复现论文。更准确地说，我完成了一条从 `MeanFlow baseline` 到 `pMF`、`Muon`、`Residual Attention` 的研究型复现链，把原本缺失的训练路径在当前 PyTorch 仓库里补齐，并在 `MNIST` 和 `CIFAR10` 上做了可比较实验。

#### Q2. 为什么先做 MNIST，最后又以 CIFAR10 为主线？

`MNIST` 适合快速验证和暴露问题，尤其能看清 baseline 与改进版本是否真的不一样。`CIFAR10` 更接近高维 pixel-space 图像问题，所以更适合作为主展示线。

#### Q3. 为什么 `pMF` 在 MNIST 上反而不如 baseline？

我当前的判断是：`MNIST` 太简单、太低维，而且卷积归纳偏置非常强。在这种设定下，baseline 本来就可能足够强，所以更复杂的目标设计不一定占优。

#### Q4. Muon 在你的项目里做了什么？

我没有把所有参数都交给 Muon，而是只让 transformer block 内的矩阵权重走 Muon，其他参数继续走 AdamW 风格更新。这样可以在保留 Muon 特点的同时，让整体训练更稳。

#### Q5. 你怎么判断当前结果还不能下最终结论？

因为 `CIFAR10` 只跑到了 `5000 step`，目前更适合看早期结构趋势，还没有更多 seed、正式指标和更长训练来支持最终判断。对研究项目来说，知道什么时候不能下结论也很重要。

### 11.4 最后一句总结

我觉得这个项目里我最大的贡献不是调参，而是构造出了一条可比较、可追溯、能从负结果中产生研究判断的实验链。

## 12. 后续补图建议

如果你后面要继续把这份文档打磨成更强的展示材料，建议优先补两类图：

1. `TensorBoard loss 曲线`
2. `固定 seed 的采样时间线`

建议的资源路径可以固定为：

- `assets/showcase/tb_cifar_loss.png`
- `assets/showcase/cifar_progress_timeline.png`

这样 README 和面试文档都可以直接复用同一套资源，而不需要重写结构。
