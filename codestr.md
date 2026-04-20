# Pixel MeanFlow 代码拆解

这份文档面向第一次阅读这个仓库源码的工程读者。目标不是把论文重讲一遍，而是把"仓库里到底有哪些代码、每个函数做什么、数据怎样在模块之间流动"讲清楚。本文档只关注 Pixel MeanFlow (pMF) 相关的代码路径。

如果你只想先抓主线，建议先看这三节：

1. `项目地图`
2. `术语表与接口面`
3. `meanflow.py 中 pMF 训练主线`

## 项目地图

### 1. 仓库结构

这个仓库有两条 pMF 实现主线：

- `meanflow.py`
  训练主线。它把模型定义、loss、采样、checkpoint、CLI 都放在一个文件里。模型为 `PmfTransformer`，loss 为 `PixelMeanFlowLoss`。
- `pmf.py -> models/pmfDiT.py`
  独立推理封装，偏向"加载既有 DiT 结构然后采样"，主要被 `evaluate.py` 使用。

配套支线：

- `fid_eval.py`、`evaluate.py`、`utils/*`
  评估与推理支线，分别服务 `meanflow.py` 训练产物的 FID 评估和独立推理链的分布式评估。
- `scripts/*`、`assets/*`、`runs/*`
  实验组织层，负责整理曲线和保存展示图。

### 2. 目录与角色

```text
pMF/
├── codestr.md                           # 这份代码拆解文档
├── meanflow.py                          # 训练主线：模型、loss、采样、checkpoint、CLI（~1139 行）
├── pmf.py                               # 独立 pMF 推理封装
├── evaluate.py                          # 面向 pmf.py 的采样/FID 入口
├── fid_eval.py                          # 面向 meanflow.py run_dir 的 FID 入口
├── README.md                            # 项目文档
├── run.md                               # 实验运行说明
├── requirements.txt                     # 依赖：torch, opencv-python, torch-fidelity>=0.3.0
├── LICENSE                              # MIT
├── .gitignore
├── models/
│   ├── pmfDiT.py                        # 独立推理链使用的 DiT 主模型（RoPE、SwiGLU、向量门控）
│   ├── embedder.py                      # 条件嵌入（TimestepEmbedder / LabelEmbedder / BottleneckPatchEmbedder）
│   └── torch_models.py                  # 基础层封装：TorchLinear / TorchEmbedding / RMSNorm / SwiGLUMlp
├── optim/
│   ├── __init__.py                      # 导出 SingleDeviceMuonWithAuxAdam
│   └── muon.py                          # Muon + AdamW 混合优化器
├── utils/
│   ├── torch_util.py                    # 设备搬运、BatchGenerator 随机批生成器
│   ├── torch_dist_util.py              # 分布式工具（NCCL/gloo 初始化、all_gather、barrier）
│   └── fidelity_wrapper.py             # 对 torch-fidelity 的薄封装，支持 .npz 参考统计
├── scripts/
│   └── plot_cifar_metrics.py           # 读取 run 指标并产出展示曲线
├── assets/                              # 项目展示图
└── runs/                                # 正式实验 run_dir 产物
```

### 3. 阅读主线

1. 先看 `codestr.md`，了解仓库分层和入口。
2. 再看 `meanflow.py` 中 pMF 相关路径：`TrainConfig -> PmfTransformer -> PixelMeanFlowLoss -> generate_pmf -> train`。
3. 然后看 `pmf.py -> models/pmfDiT.py -> evaluate.py`，这是独立推理链。
4. 最后看 `fid_eval.py + scripts/plot_cifar_metrics.py + run.md`。

### 4. 关键调用链

#### 4.1 pMF 训练调用链

```text
parse_args
  -> train
    -> 分布式初始化（检测 WORLD_SIZE，init_process_group）
    -> set_seed(seed + rank)
    -> build_datasets / build_dataloader(sampler=DistributedSampler) / cycle
    -> build_model          -> PmfTransformer
    -> build_optimizer      -> AdamW 或 Muon
    -> build_loss           -> PixelMeanFlowLoss
    -> for each step:
         -> loss_fn(model, images, labels)
         -> backward
         -> average_gradients（多卡梯度 all-reduce）
         -> optimizer.step
         -> evaluate_loss          （仅 rank 0）
         -> generate_samples       （仅 rank 0）
         -> make_checkpoint        （仅 rank 0，后接 barrier）
    -> destroy_process_group
```

#### 4.2 pMF 采样与 FID 链

```text
fid_eval.main
  -> load_run_config
  -> build_model
  -> restore_checkpoint
  -> generate_fid_samples
    -> generate_samples(variant="pmf")
      -> generate_pmf
        -> model(z_t, t, h, omega, t_min, t_max, labels)
  -> calculate_metrics
```

独立推理链：

```text
evaluate.main
  -> pixelMeanFlow(...)
  -> model.generate(...)
    -> u_fn
    -> pmfDiT.forward
```

#### 4.3 实验脚本与结果资产链

```text
scripts/plot_cifar_metrics.py
  -> 读取 runs/.../config.json + metrics.json
  -> 聚合 train/val 曲线
  -> 输出 assets/showcase/*.png
```

### 5. 对比表

#### 5.1 `PmfTransformer` vs `pmfDiT`

| 项目 | `PmfTransformer` (训练) | `pmfDiT` (推理) |
| --- | --- | --- |
| 所在文件 | `meanflow.py` | `models/pmfDiT.py` |
| 目标场景 | CIFAR-10 (32px) 训练实验 | ImageNet (256/512px) 推理 |
| 注意力 | 标准 LayerNorm + SDPA，支持 Kimi residual | RoPE + QK RMSNorm + 向量门控 |
| MLP | GELU MLP | SwiGLU MLP |
| Patch 嵌入 | 单层卷积 `PatchEmbed` | 两段卷积 `BottleneckPatchEmbedder` |
| 共同点 | 共享 backbone + u/v 双头；前缀 token 条件注入；`(x - pred) / t.clamp(min)` 速度转换 |

#### 5.2 `adamw` vs `muon`

| 优化器 | 使用位置 | 特点 |
| --- | --- | --- |
| `AdamW` | 通用 | 标准 PyTorch 优化器 |
| `Muon` | PmfTransformer | 对 shared/u/v block 中的矩阵参数走 Muon 正交化更新，其余参数走 AdamW |

## 术语表与接口面

### 1. 关键变量术语表

| 符号 | 在代码中的典型名字 | 含义 |
| --- | --- | --- |
| `x` / `x0` | `images`, `x0`, `x` | 干净图像 |
| `eps` | `eps` | 高斯噪声 |
| `z_t` | `z_t` | 时间 `t` 上的扰动样本 |
| `t` | `t`, `tau` | 当前时间 |
| `r` | `r` | 两时间量里的较早时间点 |
| `h` | `h`, `t-r` | 时间差，pMF 中显式输入模型 |
| `u` | `u` | 平均速度场 |
| `v` | `v`, `v_t`, `v_g`, `v_c`, `v_u` | 瞬时速度或其不同条件版本 |
| `omega` | `omega` | CFG 强度 |
| `t_min`, `t_max` | `t_min`, `t_max` | CFG 生效区间 |
| `fm_mask` | `fm_mask` | 用来区分 flow matching 样本与一般 two-time 样本 |
| `noise_scale` | `noise_scale` | pMF 采样噪声缩放 |

### 2. 核心接口面

#### 2.1 `TrainConfig`

`TrainConfig` 是整个 `meanflow.py` 的配置核心。pMF 相关字段：

- 路径与恢复：`workdir`, `data_root`, `resume`
- 模型选择：`dataset`, `optimizer`, `attn_impl`
- 训练调度：`seed`, `max_steps`, `batch_size`, `grad_accum_steps`, `eval_every`, `sample_every`, `save_every`, `num_workers`
- 采样参数：`sample_batch_size`, `sample_steps`, `sample_omega`, `sample_t_min`, `sample_t_max`
- 优化器超参：`lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `muon_lr`, `muon_momentum`, `muon_ns_steps`, `muon_nesterov`, `muon_weight_decay`, `muon_aux_eps`, `grad_clip`
- Transformer 结构超参：`patch_size`, `hidden_size`, `depth`, `num_heads`, `mlp_ratio`
- pMF 损失相关：`p_mean`, `p_std`, `cfg_max`, `cfg_beta`, `class_dropout_prob`, `data_proportion`, `noise_scale`, `tr_uniform`, `norm_p`, `norm_eps`, `noise_dist`
- 设备：`device`

#### 2.2 `DatasetSpec`

数据集元信息容器：`name`, `channels`, `image_size`, `num_classes`, `mean`, `std`。让模型构建、采样保存、FID 反归一化共享同一套数据规格。

#### 2.3 `RunState`

训练运行态：`step`, `best_val_loss`, `train_loss`, `val_loss`。只保存跨 step 的最小状态。

### 3. 模型 `forward` 契约

| 模型 | `forward` 形态 | 说明 |
| --- | --- | --- |
| `PmfTransformer` | `forward(x, time, h, omega, t_min, t_max, labels)` | 返回 `(u, v)` |
| `pmfDiT` | `forward(x, t, h, w, t_min, t_max, y)` | 返回 `(u, v)` |

两者都接收 `time`，但真正作为条件 token 注入的是 `h`、CFG 强度、区间和标签。`time` 主要用于把像素输出重新转换成速度量。

## `pmf.py` 独立推理封装

这个文件只有一个核心类：`pixelMeanFlow`。它做的事情可以概括为：

1. 根据字符串创建某个 `pmfDiT` 变体
2. 准备采样噪声和类别标签
3. 在每个时间步调用网络得到 `u`
4. 按 `z_t <- z_t - (t-r)u` 更新样本

### `pixelMeanFlow`

#### `pixelMeanFlow.__init__`

- 作用：
  创建独立推理模型，内部真正的网络是 `models/pmfDiT.py` 里的 `pmfDiT` 或其配置化 partial。
- 输入/输出：
  - 输入：`model_str`, `dtype`, `img_size`, `img_channels`, `num_classes`, `eval`
  - 输出：初始化后的 `nn.Module`
- 关键变量或张量形状：
  - `self.net` 是一个 `pmfDiT` 实例
  - `self.noise_scale` 由模型名字映射得到
- 内部流程：
  1. 记录模型名、dtype、图像尺寸和类别数
  2. 强制 `eval=True`，因为当前封装只支持推理模式
  3. 用 `getattr(pmfDiT, model_str)` 取到具体模型构造器
  4. 构造底层 `pmfDiT`
  5. 根据模型规格设置采样噪声缩放
- 阅读注意点：
  - `img_size` 默认 256，但 `evaluate.py` 会根据模型名把它设成 `16 * patch_size_suffix`
  - `noise_scale` 不是通过 checkpoint 读入，而是硬编码在这里

#### `pixelMeanFlow.u_fn`

- 作用：
  统一整理输入张量形状，并把请求转发给底层 `pmfDiT`。
- 输入/输出：
  - 输入：`x, t, h, omega, t_min, t_max, y`
  - 输出：`(u, v)`
- 内部流程：
  1. 读取 batch 大小
  2. 把所有标量条件 reshape 成一维 batch 向量
  3. 调用 `self.net(...)`
- 阅读注意点：
  - 返回值是 `(u, v)`，但采样时只消费第一个输出 `u`

#### `pixelMeanFlow.sample_one_step`

- 作用：
  把单个采样步包装成独立方法。
- 输入/输出：
  - 输入：`z_t, labels, i, t_steps, omega, t_min, t_max`
  - 输出：下一步样本 `z_r`
- 更新公式：`z_t - (t-r) * u`

#### `pixelMeanFlow.generate`

- 作用：
  从随机噪声开始进行一步或多步 pMF 采样。
- 输入/输出：
  - 输入：`n_sample, rng, num_steps, omega, t_min, t_max, labels=None`
  - 输出：采样后的图像张量
- 内部流程：
  1. 构造初始噪声并乘上 `noise_scale`
  2. 如果用户未提供标签，则随机采样标签
  3. 构造从 `1 -> 0` 的离散时间表
  4. 对每个时间区间：调 `u_fn` 求 `u`，用 `z_t = z_t - (t-r)u` 更新
  5. 返回最终样本
- 阅读注意点：
  - `rng` 不是标准 `torch.Generator`，而是仓库自定义的 `BatchGenerator`
  - 输出仍是训练空间内的张量，没有自动反归一化

## `models/pmfDiT.py` 独立推理链的 Transformer 主体

这个文件是独立推理版 pMF 的核心。它和 `meanflow.py` 里的 `PmfTransformer` 共享"共享 backbone + u/v 双头"的思路，但实现风格更接近生产级 DiT。

### `RoPEAttention`

- 作用：
  带 RoPE 的多头自注意力层。
- 内部流程：
  1. 线性投影出 `q/k/v`
  2. 对 `q/k` 做 `RMSNorm`
  3. 对图像 token 部分应用 `RoPE`（前缀条件 token 不旋转）
  4. 手工算 softmax attention
  5. 拼回隐藏维并做输出投影

### `TransformerBlock`

- 作用：
  `RMSNorm + RoPEAttention + RMSNorm + SwiGLU MLP`，残差分支上有可学习零初始化缩放（`attn_scale`, `mlp_scale`）。
- 更新公式：
  - `x = x + attn(norm1(x)) * attn_scale`
  - `x = x + mlp(norm2(x)) * mlp_scale`

### `FinalLayer`

- 作用：
  最终 token 到 patch 像素块的投影：`linear(norm(x))`，线性层零初始化。

### `pmfDiT`

#### `pmfDiT.__init__`

- 作用：
  搭出完整的 pMF DiT：patch embed、条件 token、共享干路、`u/v` 双头和最终输出层。
- 关键变量：
  - 条件 prefix token：`time_tokens/class_tokens/omega_tokens/t_min_tokens/t_max_tokens`
  - `shared_blocks`、`u_heads`、`v_heads`
- 内部流程：
  1. 创建 `BottleneckPatchEmbedder` 和各种条件 embedder
  2. 创建多种 learnable prefix token
  3. 构造共享主干和 `u/v` 双分支
  4. 若 `eval_mode=True`，直接把 `v` 头替换成返回全零的 lambda

#### `pmfDiT._build_sequence`

- 作用：
  把图像 patch token 和各种条件 token 拼成 transformer 输入序列。
- 拼接顺序：`[class, omega, t_min, t_max, time, image]`
- 核心思想：条件向量 + learnable token 模板

#### `pmfDiT.forward`

- 输入：`x, t, h, w, t_min, t_max, y`
- 输出：`(u, v)`
- 内部流程：
  1. 用 `_build_sequence` 准备序列
  2. 通过共享 `shared_blocks`
  3. 分叉经过 `u_heads` 与 `v_heads`
  4. 丢掉前缀 token，只保留图像 token
  5. 经过 `FinalLayer` 与 `unpatchify`
  6. 用 `(x - pred_pixels) / clamp(t)` 把图像式输出转成速度式输出
- 阅读注意点：
  - 模型并不显式用 `t` 生成条件 token，而是只在最后一步把图像预测变回速度量

### `precompute_rope_freqs` / `apply_rotary_pos_emb`

- 预计算二维 patch 网格上的 RoPE 复数频率
- 只对图像 token 施加旋转，前缀条件 token 不动

### 配置化模型

通过 `functools.partial` 预定义六种规模：`pmfDiT_B_16`, `pmfDiT_B_32`, `pmfDiT_L_16`, `pmfDiT_L_32`, `pmfDiT_H_16`, `pmfDiT_H_32`。

## `models/embedder.py` 条件嵌入与 patch 嵌入

### `TimestepEmbedder`

把标量条件先做正弦频率编码（`frequency_embedding_size` 默认 256），再送入两层 `TorchLinear + SiLU` 的 MLP。

### `LabelEmbedder`

类别 embedding 表，大小为 `num_classes + 1`（多一类用于空标签 / dropout）。

### `BottleneckPatchEmbedder`

两段卷积 patch 嵌入：先用大核卷积降到 `pca_channels`，再用 `1x1` 卷积升到 `hidden_size`。使用 Xavier-uniform 初始化。

## `models/torch_models.py` 基础层封装

### `TorchLinear`

模拟 Flax 风格初始化的线性层，支持 `scaled_variance` 和 `zeros` 两种初始化策略。

### `TorchEmbedding`

封装 `nn.Embedding`，正态分布初始化。

### `RMSNorm`

Root Mean Square Layer Normalization，带可学习缩放。

### `SwiGLUMlp`

SwiGLU 风格 MLP：`w2(silu(w1(x)) * w3(x))`，三个线性投影（`w1` + `w3` 门控，`w2` 输出）。

## `meanflow.py` 中 pMF 训练主线

这个文件是当前项目最重要的代码中心。以下只关注 pMF 相关路径。

### 1. 基础工具函数

- `cycle(loader)` — 把 DataLoader 包装成无限迭代器；当 sampler 为 `DistributedSampler` 时，在每个 epoch 边界自动调用 `set_epoch`
- `configure_runtime(...)` — 设置 cudnn/TF32/matmul 精度（尚未接入主流程）
- `set_seed(seed)` — 统一设置 Python/NumPy/PyTorch 随机种子
- `average_gradients(model, world_size)` — 遍历所有参数，对梯度做 all-reduce 后除以 world_size；单卡时为 no-op
- `is_main_process()` — 判断当前进程是否为 rank 0（或未初始化分布式）
- `to_serializable(value)` — 把 `Path`/`tuple` 转成 JSON 可写格式

### 2. Transformer 组件

### `timestep_embedding`

- 作用：
  生成正余弦时间嵌入 `[B, dim]`。
- 上游调用者：
  - `ScalarConditionEmbed.forward`

### `ScalarConditionEmbed`

- 作用：
  为标量条件构造小 MLP：先做 `timestep_embedding`，再过 MLP，得到隐藏向量。
- 上游调用者：
  - `PmfTransformer`

### `LabelConditionEmbed`

- 作用：
  构造类别 embedding 表（`num_classes + 1` 个槽位，用于空标签）。

### `PatchEmbed`

- 作用：
  把图像按 patch 切块并映射到隐藏维。`num_patches = (image_size // patch_size) ** 2`。
- 上游调用者：
  - `PmfTransformer.build_sequence`

### `rms_norm_last_dim`

- 作用：
  对最后一维做无参数 RMSNorm，服务于 `FullAttentionResidual`。

### `FullAttentionResidual`

Kimi-style 残差注意力（arXiv:2603.15031）。替代标准 `x = x + f(x)` 残差连接：

- 维护一个沿深度方向增长的 `history` 列表
- 用可学习伪查询向量对 history 做 depth 维 softmax attention
- 在每个 `TransformerBlock` 中 attention 前和 MLP 前各做一次聚合

### `TransformerAttention`

标准多头 self-attention。优先使用 `scaled_dot_product_attention`，在 JVP 场景下退回手工实现。

### `TransformerMlp`

标准两层 GELU MLP。

### `TransformerBlock`

`LayerNorm + Attention + LayerNorm + MLP` 的 block，支持两种模式：

- **naive 模式**：标准 `x + attn + mlp` 残差更新
- **residual 模式**：用 `FullAttentionResidual` 沿 depth history 聚合，不走固定残差加法

### `PmfTransformer`

#### `PmfTransformer.__init__`

- 作用：
  组装 pMF 训练版本的 transformer。
- 关键变量：
  - `null_label = num_classes`
  - 5 个前缀 token：`label/omega/t_min/t_max/h`
  - `shared_blocks`, `u_heads`, `v_heads`
  - `u_output_residual`, `v_output_residual`（仅 residual 模式）
- 内部流程：
  1. 检查深度至少为 3
  2. 创建 patch embed 和各种条件 embedder
  3. 创建 5 个 learnable prefix token
  4. 拆出共享层 (`depth - depth//2`) 和双头层 (`depth//2`)
  5. residual 模式下额外创建 `u/v` 两个输出聚合器

#### `PmfTransformer.build_sequence`

- 作用：
  构造 `[label, omega, t_min, t_max, h, image_tokens]` 序列。
- 阅读注意点：
  - `omega` 先被转换成 `1 - 1 / omega`

#### `PmfTransformer.forward`

- 输入：`x, time, h, omega, t_min, t_max, labels`
- 输出：`(u, v)`
- 内部流程：
  1. `build_sequence` 构造 token 序列
  2. naive 模式：共享 block 串行更新 `tokens`，再分别走 `u_heads` 与 `v_heads`
  3. residual 模式：共享主干扩展 `shared_history`，复制后分别经过 `u_heads/v_heads`，最后用 `output_residual` 汇聚
  4. 去掉前缀 token，只保留图像 token
  5. `norm + head + unpatchify`
  6. 用 `(x - pred_pixels) / clamp(time)` 转成 `u/v`
- 阅读注意点：
  - 真正进入 token 条件的是 `h` 等量，`time` 只用于最后从像素预测反推速度场

### 3. 扰动构造与时间采样

### `build_meanflow_corruption`

- 作用：
  构造线性插值扰动 `z_t = (1-t)x0 + t eps`
- 上游调用者：
  - `PixelMeanFlowLoss.__call__`

### 4. `PixelMeanFlowLoss`

### `PixelMeanFlowLoss.__init__`

保存 pMF 训练所需的所有分布参数、CFG 参数与自适应加权参数。

### `PixelMeanFlowLoss.sample_tr`

- 作用：
  采样 `(t, r)` 并区分哪些样本走 flow matching 特例。
- 输出：`t`, `r`, `fm_mask`
- 内部流程：
  1. 分别采样 `t` 和 `r`
  2. 如果开启 `tr_uniform`，给一部分样本改成均匀采样
  3. 按 `data_proportion` 决定多少样本强制 `r=t`（退化到 flow matching）
  4. 排序成 `t >= r`

### `PixelMeanFlowLoss.sample_cfg_scale`

采样 CFG 强度 `omega`。`cfg_beta == 1.0` 时使用指数形式，否则走 beta 形式。

### `PixelMeanFlowLoss.sample_cfg_interval`

为每个样本采样 CFG 生效区间 `[t_min, t_max]`。`fm_mask=True` 的样本固定成 `[0,1]`。

### `PixelMeanFlowLoss.v_cond_fn`

调用模型拿到带条件的 `v` 头输出，`h` 固定成零，区间固定成 `[0,1]`。

### `PixelMeanFlowLoss.v_fn`

一次前向同时得到条件版 `v_c` 和无条件版 `v_u`：把 batch 拼成双倍，一半真实标签 + omega，一半 null 标签 + omega=1。这是 classifier-free guidance 训练侧的批内并行技巧。

### `PixelMeanFlowLoss.guidance_fn`

构造目标指导速度 `v_g`。对 `fm_mask` 样本使用全程 CFG 公式，其余样本只在区间内启用 CFG。

### `PixelMeanFlowLoss.cond_drop`

按概率做类别条件 dropout：被丢弃的样本标签改成 `null_label`，目标速度退回 `v_t`。

### `PixelMeanFlowLoss.adaptive_weight`

自适应加权，降低大损失样本的直接主导性。

### `PixelMeanFlowLoss.__call__`

- 输入：`net`, `images`, `labels`
- 输出：标量 loss
- 内部流程：
  1. 检查必须有标签
  2. 采样 `t/r/fm_mask`
  3. 采样噪声并构造 `z_t`
  4. 由 `z_t` 和 `x` 计算当前真实速度 `v_t`
  5. 采样 CFG 区间和强度
  6. 用 `guidance_fn` 构造指导速度 `v_g`
  7. 做条件 dropout
  8. 定义 `warped_u_fn`，让模型返回 `(u, v)`
  9. 对 `warped_u_fn` 做 JVP，得到 `(u, v)` 和 `du_dt`
  10. 构造 `v_compound = u + (t-r) * du_dt`
  11. 分别计算 `u` 分支和 `v` 分支的损失
  12. 对两项损失做自适应加权后求均值
- 核心含义：
  - 这里把 pMF 的"图像式预测 + 速度空间监督 + CFG 训练"全部串起来了

### 5. 采样函数

### `build_sample_labels`

为 pMF 采样构造循环标签 `[0,1,2,...] % num_classes`。

### `generate_pmf`

- 作用：
  用离散时间表执行 pMF 采样。
- 内部流程：
  1. 初始噪声乘上 `model.noise_scale`
  2. 构造 `1 -> 0` 的时间表
  3. 每一步：扩展 `t/r`，构造固定 `omega/t_min/t_max`，调模型拿 `u`，更新 `z_t = z_t - (t-r)u`

### `generate_samples`

统一采样入口，直接调用 `generate_pmf`。

### 6. 数据、模型、优化器、loss 工厂

### `get_dataset_spec`

返回 `cifar10` 的标准规格（`DatasetSpec`）。

### `build_datasets`

构造训练集、验证集和对应数据规格。CIFAR-10 使用随机水平翻转，归一化到 `[-1, 1]`。

### `build_dataloader`

为给定数据集创建 `DataLoader`，带固定种子的 `torch.Generator`。支持可选的 `sampler` 参数，多卡训练时传入 `DistributedSampler` 以实现数据分片。

### `build_model`

直接返回 `PmfTransformer`。

### `build_optimizer`

- `adam` / `adamw`：直接返回标准优化器
- `muon`：按参数名和张量维度拆成 `muon_params`（shared/u/v block 中的矩阵参数）与 `adamw_params`（其余参数），返回 `SingleDeviceMuonWithAuxAdam`

### `build_loss`

直接返回 `PixelMeanFlowLoss`。

### 7. 运行目录、样本保存与 checkpoint

- `get_run_dir` — 按时间戳创建新目录，或 resume 时回到旧目录
- `save_json` — UTF-8 写 JSON
- `save_sample_grid` — 反归一化后存成网格图
- `evaluate_loss` — 用少量 batch 估计验证损失
- `make_checkpoint` — 打包模型、优化器、配置、训练状态与 RNG 状态
- `restore_checkpoint` — 恢复模型、优化器和运行态

### 8. CLI 与训练循环

### `train`

- 作用：
  执行一次完整训练并返回 run 目录。支持单卡和多卡分布式训练。
- 分布式策略：
  由于 `PixelMeanFlowLoss` 使用 `torch.func.jvp`（前向模式自动微分），与 DDP 的 backward hook 不兼容，因此不使用 `DistributedDataParallel` 封装模型，而是在 backward 完成后手动对梯度做 all-reduce。
- 内部流程：
  1. 检测 `WORLD_SIZE` 环境变量，初始化分布式进程组（`nccl` 后端）
  2. 按 rank 偏移随机种子 `set_seed(seed + rank)`，确保各卡数据/噪声不同
  3. 仅 rank 0 创建目录，之后 barrier 同步
  4. 构造数据（多卡时使用 `DistributedSampler`）、模型、优化器、loss
  5. 可选 resume checkpoint，resume 后重新偏移种子
  6. 构造固定采样噪声和标签
  7. 主循环：梯度累积 -> `average_gradients` -> clip grad -> optimizer.step
  8. 验证/采样/保存仅在 rank 0 执行，checkpoint 保存后加 barrier
  9. 训练结束后调用 `destroy_process_group`
- 启动方式：
  - 单卡：`python meanflow.py ...`（与之前完全兼容）
  - 多卡：`torchrun --nproc_per_node=N meanflow.py ...`
  - `--batch-size` 为每张卡的 batch size，有效 batch = `batch_size × world_size × grad_accum_steps`

## `optim/muon.py` Muon 优化器实现

### `zeropower_via_newtonschulz5`

用 5 阶 Newton-Schulz 迭代近似梯度矩阵的零次幂归一化。在 `bfloat16` 上运算，硬编码系数 `(a=3.4445, b=-4.7750, c=2.0315)`。

### `muon_update`

对单个参数执行 Muon 更新：标准 momentum buffer -> 可选 Nesterov -> Newton-Schulz 正交化 -> 按长宽比缩放。

### `adam_update`

标准 bias-corrected Adam 更新。

### `SingleDeviceMuonWithAuxAdam`

混合优化器：同一实例内管理 Muon 组（transformer 矩阵参数）和 AdamW 组（embeddings/biases/norms），统一在 `.step()` 中执行。

## `utils/torch_util.py` 设备与随机数辅助

- `seed(seed)` — 设置 PyTorch 随机种子
- `device_get(arr)` — GPU tensor -> CPU numpy
- `device_put(arr)` — 搬到当前分布式设备
- `BatchGenerator` — 为 batch 中每个样本维护独立 `torch.Generator`，复现 JAX/EDM 的"每样本独立随机源"语义
- `tree_map(f, tree)` — 递归 map 嵌套结构

## `utils/torch_dist_util.py` 分布式辅助

服务 `evaluate.py` 的分布式采样/FID 流程：

- `initialize()` — 初始化 NCCL/gloo 进程组
- `process_index()` / `process_count()` / `local_device()` — 进程标识
- `all_gather(d)` — tensor/dict all-gather
- `barrier()` — 同步屏障
- `print0()` — 只在 rank 0 打印

## `utils/fidelity_wrapper.py` FID 封装

补上 `torch-fidelity` 原生不支持的"URL 指向 `.npz` 参考统计文件"能力：

- `url_to_path(url_or_path)` — URL 则下载到本地缓存，否则原样返回
- `calculate_metrics(...)` — 保留 `torch-fidelity` 主逻辑，FID 路径从 `.npz` 读 `mu/sigma`

## 评估、运行与可视化脚本

### `evaluate.py`

面向独立推理链 `pixelMeanFlow + pmfDiT`。支持 `sample`（生成拼图）和 `evaluate`（50K 图 + FID/IS）两种模式。分布式流程：按 rank 分样本、`BatchGenerator` 采样、逐张写 PNG、rank 0 算指标。

### `fid_eval.py`

面向 `meanflow.py` 训练产物。读取 `run_dir/config.json` 和 checkpoint，调用 `build_model` + `restore_checkpoint` + `generate_samples` 生成图像，然后用 `calculate_metrics` 算 FID。

### `scripts/plot_cifar_metrics.py`

读取 CIFAR10 run 的 `metrics.json/config.json`，EMA 平滑训练损失，绘制 train/val 对比图。

## 一句话总结

- `meanflow.py` 是训练平台，pMF 路径使用 `PmfTransformer` + `PixelMeanFlowLoss`
- `pmf.py + pmfDiT.py` 是独立推理封装
- 两条线共享核心思路：共享 backbone + u/v 双头、前缀 token 条件注入、图像式预测 + 速度空间监督
