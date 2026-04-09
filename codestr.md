# Pixel MeanFlow 代码拆解

这份文档面向第一次阅读这个仓库源码的工程读者。目标不是把论文重讲一遍，而是把“仓库里到底有哪些代码、每个函数做什么、数据怎样在模块之间流动”讲清楚。

如果你只想先抓主线，建议先看这四节：

1. `项目地图`
2. `术语表与接口面`
3. `meanflow.py 训练实验主线`
4. `pMF 独立推理链`

如果你想配合图一起看：

- `meanflow_pipeline.excalidraw`
  关注 `meanflow.py` 里的训练循环、loss 和采样链。
- `project_structure.excalidraw`
  关注仓库分层、入口脚本和实验资产之间的关系。

## 项目地图

### 1. 仓库里有两条实现主线、两条配套支线

这个仓库不是一个单一入口的工程，而是“四层角色”并存：

- `pMF/meanflow.py`
  这是当前实验主线。它把训练、模型定义、loss、采样、checkpoint、CLI 都放在一个文件里，支持：
  - `variant=meanflow`
  - `variant=pmf`
  - `backbone=unet`
  - `backbone=transformer`
- `pMF/pmf.py -> pMF/models/pmfDiT.py`
  这是一个独立的 pMF 推理封装，偏向“加载既有 DiT 结构然后采样”，主要被 `pMF/evaluate.py` 使用。
- `pMF/fid_eval.py`、`pMF/evaluate.py`、`pMF/utils/*`
  这是评估与推理支线。它们分别服务：
  - `meanflow.py` 训练产物的 FID 评估
  - 独立 `pmf.py` 推理链的采样 / 分布式评估
- `pMF/scripts/*`、`pMF/assets/showcase/*`、`pMF/runs/*`
  这是实验组织层。它不定义模型本身，但负责批量跑实验、整理曲线和保存展示图。

可以把它理解成：

- `meanflow.py` 是“实验平台版本”
- `pmf.py + pmfDiT.py` 是“独立推理版本”
- `fid_eval.py / evaluate.py` 是“评估和采样入口”
- `scripts/ + runs/ + assets/showcase/` 是“实验资产层”

### 2. 目录与角色

```text
pixelmeanflow/
├── codestr.md                           # 这份代码拆解文档
├── generate_meanflow_diagram.py         # 生成训练流程图
├── meanflow_pipeline.excalidraw         # 训练流程图产物
├── generate_project_structure_diagram.py # 生成项目结构图
├── project_structure.excalidraw         # 项目结构图产物
├── docs/                                # 论文与学习笔记，不是运行时代码
├── pMF/
│   ├── meanflow.py                      # 训练主线：模型、loss、采样、checkpoint、CLI
│   ├── pmf.py                           # 独立 pMF 推理封装
│   ├── evaluate.py                      # 面向 pmf.py 的采样/FID 入口
│   ├── fid_eval.py                      # 面向 meanflow.py run_dir 的 FID 入口
│   ├── run.md                           # 实验运行说明
│   ├── requirements.txt                 # pMF 子项目依赖
│   ├── models/
│   │   ├── pmfDiT.py                    # 独立推理链使用的 DiT 主模型
│   │   ├── embedder.py                  # 各类条件嵌入与 patch embed
│   │   └── torch_models.py              # 基础层封装：Linear / Embedding / RMSNorm / SwiGLU
│   ├── optim/
│   │   └── muon.py                      # Muon + AdamW 混合优化器
│   ├── utils/
│   │   ├── torch_util.py                # 设备搬运、随机批生成器
│   │   ├── torch_dist_util.py           # 分布式工具
│   │   └── fidelity_wrapper.py          # 对 torch-fidelity 的薄封装
│   ├── scripts/
│   │   ├── run_cifar10_suite.sh         # 批量启动 CIFAR10 训练实验
│   │   ├── plot_cifar_metrics.py        # 读取 run 指标并产出展示曲线
│   │   └── plot_fixcheck_metrics.py     # 对 fixcheck 实验做对比可视化
│   ├── assets/
│   │   ├── teaser.png                   # 论文/项目展示图
│   │   └── showcase/                    # 由脚本生成或整理的实验图
│   ├── runs/                            # 正式实验 run_dir 产物
│   ├── tmp_stage*/                      # 临时 smoke / stage run_dir
│   ├── data/                            # 本地数据缓存
│   └── tests/
│       └── test_muon.py                 # Muon 相关单测
└── paper/                               # 论文 PDF
```

按角色可以再压缩成三层：

- 文档与图表层：
  `codestr.md`、`docs/`、`generate_*diagram.py`、`*.excalidraw`
- 源码层：
  `meanflow.py`、`pmf.py`、`models/`、`optim/`、`utils/`、评估入口脚本
- 运行与实验资产层：
  `run.md`、`scripts/`、`assets/showcase/`、`runs/`、`tmp_stage*/`、`data/`

其中最后一层大多是“运行结果或实验配套资源”，不是核心算法实现本体。

### 3. 项目结构阅读主线

第一次读这个仓库时，推荐按下面的顺序走：

1. 先看 `codestr.md + project_structure.excalidraw`
   目的是先知道仓库分层、入口文件和实验资产放在哪里。
2. 再看 `pMF/meanflow.py + meanflow_pipeline.excalidraw`
   这是当前最完整的训练实验主线。
3. 然后看 `pMF/pmf.py -> pMF/models/pmfDiT.py -> pMF/evaluate.py`
   这是独立推理链，和 `meanflow.py` 共享论文背景，但代码组织不同。
4. 最后看 `pMF/fid_eval.py + pMF/scripts/*.py + pMF/run.md`
   这一层负责把 run_dir 变成 FID、曲线图和可复现实验命令。

### 4. 四条关键调用链

#### 4.1 训练调用链

```text
parse_args
  -> train
    -> set_seed
    -> build_datasets / build_dataloader / cycle
    -> build_model
    -> build_optimizer
    -> build_loss
    -> for each step:
         -> loss_fn(model, images, labels)
         -> backward
         -> optimizer.step
         -> evaluate_loss
         -> generate_samples
         -> save_sample_grid
         -> make_checkpoint / save_json
```

#### 4.2 MeanFlow 采样链

```text
generate_samples(variant="meanflow")
  -> generate_meanflow
    -> model(noise_x, tau=1, h=1)
    -> noise_x - predicted_u
```

这是单步采样。`meanflow` 变体并不做多步时间积分。

#### 4.3 pMF 采样与 FID 链

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

独立推理链则是：

```text
evaluate.main
  -> pixelMeanFlow(...)
  -> model.generate(...)
    -> u_fn
    -> pmfDiT.forward
```

#### 4.4 实验脚本与结果资产链

```text
scripts/run_cifar10_suite.sh
  -> 多次 python meanflow.py ...
  -> pMF/runs/<experiment>/<timestamp>/
     -> config.json / metrics.json / samples/ / checkpoints/

scripts/plot_cifar_metrics.py
scripts/plot_fixcheck_metrics.py
  -> 读取 runs/.../config.json + metrics.json
  -> 聚合 train/val 曲线
  -> 输出 assets/showcase/*.png
```

### 5. 三组对比表

#### 5.1 `meanflow` vs `pmf`

| 项目 | `meanflow` | `pmf` |
| --- | --- | --- |
| 训练目标 | 直接学平均速度 `u` | 学 `u` 和辅助 `v`，并做 CFG 相关训练 |
| 标签依赖 | 可无标签 | 必须有类别标签 |
| 采样 | `generate_meanflow` 单步 | `generate_pmf` 多步或一步 |
| 模型候选 | `Unet` 或 `MeanFlowTransformer` | `PmfTransformer` |
| loss 类 | `MeanFlowLoss` | `PixelMeanFlowLoss` |

#### 5.2 `Unet` vs `MeanFlowTransformer` vs `PmfTransformer`

| 模型 | 角色 | 条件方式 | 输出 |
| --- | --- | --- | --- |
| `Unet` | `meanflow` 的卷积骨干 | 时间嵌入加到 block | 预测 `u` |
| `MeanFlowTransformer` | `meanflow` 的 transformer 骨干 | 把时间/步长条件加到 patch token | 预测 `u` |
| `PmfTransformer` | `pmf` 训练主模型 | 前缀 token 注入 `h/omega/t_min/t_max/label` | 同时预测 `u` 和 `v` |

#### 5.3 `adamw` vs `muon`

| 优化器 | 使用位置 | 特点 |
| --- | --- | --- |
| `Adam` / `AdamW` | 通用 | 标准 PyTorch 优化器 |
| `Muon` | 仅 `variant=pmf` | 对 transformer 主矩阵参数走 Muon，其余参数走 Adam 风格更新 |

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

`TrainConfig` 是整个 `meanflow.py` 的配置核心。字段可以分成几类：

- 路径与恢复
  - `workdir`
  - `data_root`
  - `resume`
- 数据与模型选择
  - `dataset`
  - `backbone`
  - `variant`
  - `optimizer`
  - `attn_impl`
- 训练调度
  - `seed`
  - `max_steps`
  - `batch_size`
  - `grad_accum_steps`
  - `eval_every`
  - `sample_every`
  - `save_every`
  - `num_workers`
- 采样参数
  - `sample_batch_size`
  - `sample_steps`
  - `sample_omega`
  - `sample_t_min`
  - `sample_t_max`
- 优化器超参
  - `lr`
  - `beta1`
  - `beta2`
  - `eps`
  - `weight_decay`
  - `muon_lr`
  - `muon_momentum`
  - `muon_ns_steps`
  - `muon_nesterov`
  - `muon_weight_decay`
  - `muon_aux_eps`
  - `grad_clip`
- U-Net / Transformer 结构超参
  - `model_dim`
  - `dim_mults`
  - `convnext_mult`
  - `patch_size`
  - `hidden_size`
  - `depth`
  - `num_heads`
  - `mlp_ratio`
- MeanFlow / pMF 损失相关
  - `t_min`
  - `p_mean`
  - `p_std`
  - `cfg_max`
  - `cfg_beta`
  - `class_dropout_prob`
  - `data_proportion`
  - `noise_scale`
  - `tr_uniform`
  - `norm_p`
  - `norm_eps`
  - `noise_dist`
- 设备
  - `device`

#### 2.2 `DatasetSpec`

这是数据集元信息的统一容器，字段包括：

- `name`
- `channels`
- `image_size`
- `num_classes`
- `mean`
- `std`

它的作用是让模型构建、采样保存、FID 反归一化都能共享同一套数据规格。

#### 2.3 `RunState`

训练运行态：

- `step`
- `best_val_loss`
- `train_loss`
- `val_loss`

它只保存训练时需要跨 step 保留的最小状态，不包含模型参数本身。

### 3. 模型 `forward` 契约

| 模型 | `forward` 形态 | 说明 |
| --- | --- | --- |
| `Unet` | `forward(x, time, h=None)` | 预测 `u` |
| `MeanFlowTransformer` | `forward(x, time, h=None)` | 预测 `u` |
| `PmfTransformer` | `forward(x, time, h, omega, t_min, t_max, labels)` | 返回 `(u, v)` |
| `pmfDiT` | `forward(x, t, h, w, t_min, t_max, y)` | 返回 `(u, v)` |

注意：

- `PmfTransformer` 和 `pmfDiT` 都接收 `time`，但真正作为条件 token 注入的是 `h`、CFG 强度、区间和标签。
- `time` 主要用于把像素输出重新转换成速度量。

## `pMF/pmf.py` 独立推理封装

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
- 上游调用者：
  - `evaluate.main`
  - 用户自己直接实例化
- 下游依赖：
  - `pmfDiT` 系列 partial：`pmfDiT_B_16` 等
- 阅读注意点：
  - 这里的 `img_size` 默认 256，但 `evaluate.py` 会根据模型名把它设成 `16 * patch_size_suffix`
  - `noise_scale` 不是通过 checkpoint 读入，而是硬编码在这里

#### `pixelMeanFlow.u_fn`

- 作用：
  统一整理输入张量形状，并把请求转发给底层 `pmfDiT`。
- 输入/输出：
  - 输入：`x, t, h, omega, t_min, t_max, y`
  - 输出：`(u, v)`
- 关键变量或张量形状：
  - `x`: `[B, C, H, W]`
  - `t/h/omega/t_min/t_max`: 会被 reshape 成 `[B]`
- 内部流程：
  1. 读取 batch 大小
  2. 把所有标量条件 reshape 成一维 batch 向量
  3. 调用 `self.net(...)`
- 上游调用者：
  - `sample_one_step`
  - `generate`
- 下游依赖：
  - `pmfDiT.forward`
- 阅读注意点：
  - 返回值是 `(u, v)`，但采样时只消费第一个输出 `u`

#### `pixelMeanFlow.sample_one_step`

- 作用：
  把单个采样步包装成独立方法。
- 输入/输出：
  - 输入：`z_t, labels, i, t_steps, omega, t_min, t_max`
  - 输出：下一步样本 `z_r`
- 关键变量或张量形状：
  - `t = t_steps[i]`
  - `r = t_steps[i + 1]`
  - 更新公式：`z_t - (t-r) * u`
- 内部流程：
  1. 取当前时间 `t` 和下一个时间 `r`
  2. 把标量条件扩成 batch 形状
  3. 用 `u_fn` 拿到 `u`
  4. 应用欧拉式更新
- 上游调用者：
  - 当前代码中没有被 `generate` 复用，属于可复用但未被用到的辅助方法
- 下游依赖：
  - `u_fn`
- 阅读注意点：
  - 逻辑与 `generate` 里的循环体几乎一致，阅读时可以把它当成采样步的简化版说明

#### `pixelMeanFlow.generate`

- 作用：
  从随机噪声开始进行一步或多步 pMF 采样。
- 输入/输出：
  - 输入：`n_sample, rng, num_steps, omega, t_min, t_max, labels=None`
  - 输出：采样后的图像张量
- 关键变量或张量形状：
  - 初始噪声 `z_t`: `[B, C, H, W]`
  - `t_steps`: `[num_steps + 1]`
  - `y`: `[B]`
- 内部流程：
  1. 构造初始噪声并乘上 `noise_scale`
  2. 如果用户未提供标签，则随机采样标签
  3. 构造从 `1 -> 0` 的离散时间表
  4. 把 `omega/t_min/t_max` 规范成 tensor
  5. 对每个时间区间：
     1. 扩展 batch 版 `t/r/omega/t_min/t_max`
     2. 调 `u_fn` 求 `u`
     3. 用 `z_t = z_t - (t-r)u` 更新
  6. 返回最终样本
- 上游调用者：
  - `evaluate.run_evaluate`
  - `evaluate.main` 的 `sample` 分支
- 下游依赖：
  - `u_fn`
  - `pmfDiT.forward`
- 阅读注意点：
  - 这是真正的采样主循环
  - `rng` 不是标准 `torch.Generator`，而是仓库自定义的 `BatchGenerator`
  - 这里没有自动反归一化，输出仍是训练空间内的张量

## `pMF/models/pmfDiT.py` 独立推理链的 Transformer 主体

这个文件是独立推理版 pMF 的核心。它和 `meanflow.py` 里的 `PmfTransformer` 很像，但实现风格更接近“单独整理过的 DiT 模块”。

### `unsqueeze`

- 作用：
  对 `torch.unsqueeze` 做一层薄封装。
- 输入/输出：
  - 输入：`t`, `dim`
  - 输出：在指定维度扩出的张量
- 关键变量或张量形状：
  - 常用于把 `[B, D]` 变成 `[B, 1, D]`
- 内部流程：
  - 直接调用 `t.unsqueeze(dim)`
- 上游调用者：
  - `_build_sequence`
  - `apply_rotary_pos_emb`
- 下游依赖：
  - 无
- 阅读注意点：
  - 逻辑很薄，主要是为了和原实现风格保持一致

### `RoPEAttention`

#### `RoPEAttention.__init__`

- 作用：
  定义带 RoPE 的多头自注意力层。
- 输入/输出：
  - 输入：`hidden_size`, `num_heads`, 初始化相关参数
  - 输出：模块初始化
- 关键变量或张量形状：
  - `head_dim = hidden_size // num_heads`
  - `q_proj/k_proj/v_proj/out_proj` 都是 `TorchLinear`
- 内部流程：
  1. 创建 Q/K/V/O 投影
  2. 记录 head 维度
  3. 为 Q/K 单独配 `RMSNorm`
- 上游调用者：
  - `TransformerBlock`
- 下游依赖：
  - `TorchLinear`
  - `RMSNorm`

#### `RoPEAttention.forward`

- 作用：
  完成一次带旋转位置编码的 self-attention。
- 输入/输出：
  - 输入：`x`, `rope_freqs`
  - 输出：attention 输出
- 关键变量或张量形状：
  - `x`: `[B, S, H]`
  - `q/k/v`: `[B, S, heads, head_dim]`
  - `attn_weights`: `[B, heads, S, S]`
- 内部流程：
  1. 线性投影出 `q/k/v`
  2. 对 `q/k` 做 `RMSNorm`
  3. 对图像 token 部分应用 `RoPE`
  4. 手工算 softmax attention
  5. 拼回隐藏维并做输出投影
- 上游调用者：
  - `TransformerBlock.forward`
- 下游依赖：
  - `apply_rotary_pos_emb`
- 阅读注意点：
  - 作者手工实现 attention，是为了更贴近原 JAX 实现

### `TransformerBlock`

#### `TransformerBlock.__init__`

- 作用：
  组合 `RMSNorm + attention + RMSNorm + SwiGLU MLP`，并在残差分支上放可学习缩放。
- 输入/输出：
  - 输入：隐藏维、头数、MLP 比例和初始化参数
  - 输出：模块初始化
- 关键变量或张量形状：
  - `attn_scale`、`mlp_scale` 初值全零
- 内部流程：
  1. 建 `norm1 + attn`
  2. 建 `norm2 + mlp`
  3. 建两个残差门控参数
- 上游调用者：
  - `pmfDiT`
- 下游依赖：
  - `RoPEAttention`
  - `SwiGLUMlp`

#### `TransformerBlock.forward`

- 作用：
  进行一层 transformer block 更新。
- 输入/输出：
  - 输入：`x`, `rope_freqs`
  - 输出：更新后的 token
- 关键变量或张量形状：
  - attention 和 MLP 输出都会按逐通道 gate 缩放
- 内部流程：
  1. `x = x + attn(norm1(x)) * attn_scale`
  2. `x = x + mlp(norm2(x)) * mlp_scale`
- 上游调用者：
  - `pmfDiT.forward`
- 下游依赖：
  - `RoPEAttention.forward`
  - `SwiGLUMlp.forward`

### `FinalLayer`

#### `FinalLayer.__init__`

- 作用：
  定义最终 token 到 patch 像素块的投影层。
- 输入/输出：
  - 输入：`hidden_size`, `patch_size`, `out_channels`
  - 输出：模块初始化
- 关键变量或张量形状：
  - 输出维度为 `patch_size * patch_size * out_channels`
- 内部流程：
  1. 建一个 `RMSNorm`
  2. 建一个零初始化 `TorchLinear`
- 上游调用者：
  - `pmfDiT`
- 下游依赖：
  - `TorchLinear`
  - `RMSNorm`

#### `FinalLayer.__call__`

- 作用：
  对 token 做标准化并映射回 patch 像素空间。
- 输入/输出：
  - 输入：token 序列
  - 输出：每个 token 对应的 patch 向量
- 关键变量或张量形状：
  - 输出形状约为 `[B, num_patches, patch_area * channels]`
- 内部流程：
  - `linear(norm(x))`
- 上游调用者：
  - `pmfDiT.forward`
- 下游依赖：
  - 无
- 阅读注意点：
  - 这里覆写了 `__call__` 而不是写 `forward`，阅读时要留意

### `pmfDiT`

#### `pmfDiT.__init__`

- 作用：
  搭出完整的 pMF DiT：patch embed、条件 token、共享干路、`u/v` 双头和最终输出层。
- 输入/输出：
  - 输入：图像尺寸、patch、大模型宽深、类别数、双头深度等
  - 输出：模块初始化
- 关键变量或张量形状：
  - `time_tokens/class_tokens/omega_tokens/t_min_tokens/t_max_tokens`
  - `shared_blocks`
  - `u_heads`
  - `v_heads`
  - `prefix_tokens`
- 内部流程：
  1. 创建图像 patch embedder
  2. 创建多个条件 embedder
  3. 创建多种 learnable prefix token
  4. 计算总 token 数并初始化位置编码
  5. 构造共享主干和 `u/v` 双分支
  6. 构造最终输出层
  7. 若 `eval_mode=True`，直接把 `v` 头替换成返回全零的 lambda
- 上游调用者：
  - `pixelMeanFlow.__init__`
  - 文件末尾 partial 配置
- 下游依赖：
  - `BottleneckPatchEmbedder`
  - `TimestepEmbedder`
  - `LabelEmbedder`
  - `TransformerBlock`
  - `FinalLayer`
- 阅读注意点：
  - 这里显式区分共享层和双头层，体现了 “共享 backbone + 双任务头” 的结构

#### `pmfDiT.unpatchify`

- 作用：
  把 patch token 还原回图像。
- 输入/输出：
  - 输入：patch 输出 `[B, T, patch_area * C]`
  - 输出：图像 `[B, C, H, W]`
- 关键变量或张量形状：
  - `T` 必须是平方数
- 内部流程：
  1. 把 token reshape 回二维网格
  2. 用 `einsum` 调整维度顺序
  3. 拼回整图
- 上游调用者：
  - `pmfDiT.forward`
- 下游依赖：
  - 无

#### `pmfDiT._build_sequence`

- 作用：
  把图像 patch token 和各种条件 token 拼成 transformer 输入序列。
- 输入/输出：
  - 输入：`x, h, w, t_min, t_max, y`
  - 输出：完整 token 序列
- 关键变量或张量形状：
  - 图像 token 在后，prefix token 在前
- 内部流程：
  1. embed 图像
  2. embed 时间差、CFG 强度、CFG 区间和标签
  3. 把 learnable token 与条件向量相加
  4. 拼接成 `[class, omega, t_min, t_max, time, image]`
  5. 加上位置编码
- 上游调用者：
  - `pmfDiT.forward`
- 下游依赖：
  - `x_embedder`
  - 各类 embedder
- 阅读注意点：
  - 这里条件注入的核心思想是“条件向量 + learnable token 模板”

#### `pmfDiT.forward`

- 作用：
  完成独立推理版 pMF 的前向传播，输出 `u` 和 `v`。
- 输入/输出：
  - 输入：`x, t, h, w, t_min, t_max, y`
  - 输出：`(u, v)`
- 关键变量或张量形状：
  - `seq`: `[B, prefix + patches, hidden]`
  - `u_tokens/v_tokens`: 去掉 prefix 后的图像 token
  - `u/v`: `[B, C, H, W]`
- 内部流程：
  1. 用 `_build_sequence` 准备序列
  2. 通过共享 `shared_blocks`
  3. 复制到 `u_seq` 和 `v_seq`
  4. 分别经过 `u_heads` 与 `v_heads`
  5. 丢掉前缀 token，只保留图像 token
  6. 经过最终线性层与 `unpatchify`
  7. 用 `(x - pred_pixels) / clamp(t)` 把图像式输出转成速度式输出
- 上游调用者：
  - `pixelMeanFlow.u_fn`
- 下游依赖：
  - `_build_sequence`
  - `TransformerBlock.forward`
  - `FinalLayer`
  - `unpatchify`
- 阅读注意点：
  - 这个模型并不显式用 `t` 生成条件 token，而是只在最后一步把图像预测变回速度量
- 对应方法含义：
  - 这是“先预测更像图像的量，再转回速度空间”的代码落点

### `precompute_rope_freqs`

- 作用：
  预计算二维 patch 网格上的 RoPE 复数频率。
- 输入/输出：
  - 输入：`dim`, `seq_len`, `theta`
  - 输出：复数张量 `freqs_cis`
- 关键变量或张量形状：
  - 输出形状与图像 patch token 数匹配
- 内部流程：
  1. 取二维网格边长 `T = sqrt(seq_len)`
  2. 分别为高和宽构造频率
  3. 拼成二维复数旋转表
- 上游调用者：
  - `pmfDiT.__init__`
- 下游依赖：
  - `apply_rotary_pos_emb`

### `apply_rotary_pos_emb`

- 作用：
  只对最后的图像 token 施加 rotary embedding，保留前缀条件 token 不旋转。
- 输入/输出：
  - 输入：`x`, `freqs_cis`
  - 输出：旋转后的 `x`
- 关键变量或张量形状：
  - 前缀 token 不动
  - 最后 `T` 个图像 token 会旋转
- 内部流程：
  1. 把最后一维按两两配对转成复数
  2. 扩展 `freqs_cis` 到 batch/head 维
  3. 只更新最后的图像 token 部分
  4. 再转回实数表示
- 上游调用者：
  - `RoPEAttention.forward`
- 下游依赖：
  - `unsqueeze`

### 配置化模型：`pmfDiT_B_16` 等

- 作用：
  通过 `functools.partial` 预定义一组常用规模。
- 成员：
  - `pmfDiT_B_16`
  - `pmfDiT_B_32`
  - `pmfDiT_L_16`
  - `pmfDiT_L_32`
  - `pmfDiT_H_16`
  - `pmfDiT_H_32`
- 阅读注意点：
  - `pixelMeanFlow.__init__` 正是通过模型名字符串去取这些 partial

## `pMF/models/embedder.py` 条件嵌入与 patch 嵌入

### `TimestepEmbedder`

#### `TimestepEmbedder.__init__`

- 作用：
  把一个标量条件先做正弦频率编码，再送入 MLP。
- 输入/输出：
  - 输入：隐藏维、频率编码维、初始化参数
  - 输出：模块初始化
- 关键变量或张量形状：
  - `frequency_embedding_size` 默认 256
- 内部流程：
  1. 创建两层 `TorchLinear + SiLU` 的 MLP
- 上游调用者：
  - `pmfDiT.__init__`
- 下游依赖：
  - `TorchLinear`

#### `TimestepEmbedder.timestep_embedding`

- 作用：
  把标量时间编码成正余弦向量。
- 输入/输出：
  - 输入：`t`, `dim`, `max_period`
  - 输出：`[B, dim]`
- 关键变量或张量形状：
  - 如果 `dim` 是奇数，会补一位零
- 内部流程：
  1. 构造指数频率
  2. 计算 `cos` 与 `sin`
  3. 必要时补零
- 上游调用者：
  - `TimestepEmbedder.forward`

#### `TimestepEmbedder.forward`

- 作用：
  生成最终的时间条件向量。
- 输入/输出：
  - 输入：`t`
  - 输出：隐藏维向量
- 内部流程：
  1. `timestep_embedding`
  2. MLP 投影
- 上游调用者：
  - `pmfDiT._build_sequence`

### `LabelEmbedder`

#### `LabelEmbedder.__init__`

- 作用：
  为类别标签构造 embedding 表。
- 输入/输出：
  - 输入：类别数、隐藏维、初始化参数
  - 输出：模块初始化
- 关键变量或张量形状：
  - embedding 表大小为 `num_classes + 1`
- 阅读注意点：
  - 多出来的一类通常给“空标签”或 dropout 标签使用

#### `LabelEmbedder.forward`

- 作用：
  按标签索引 embedding。
- 输入/输出：
  - 输入：`labels`
  - 输出：对应类别向量

### `BottleneckPatchEmbedder`

#### `BottleneckPatchEmbedder.__init__`

- 作用：
  用“两段卷积”把图像转成 patch token。
- 输入/输出：
  - 输入：图像尺寸、初始 patch 大小、瓶颈通道数、输入通道数、隐藏维
  - 输出：模块初始化
- 关键变量或张量形状：
  - `proj1`: `[C_in -> pca_channels]`
  - `proj2`: `[pca_channels -> hidden_size]`
- 内部流程：
  1. 计算 patch 网格大小
  2. 创建两个卷积
  3. 手工按 linear 方式初始化两个卷积权重
- 上游调用者：
  - `pmfDiT.__init__`
- 下游依赖：
  - `_init_img_size`

#### `BottleneckPatchEmbedder._init_img_size`

- 作用：
  由图像尺寸和 patch 大小推导网格信息。
- 输入/输出：
  - 输入：`img_size`
  - 输出：`img_size`, `grid_size`, `num_patches`

#### `BottleneckPatchEmbedder.forward`

- 作用：
  把图像张量变成 token 序列。
- 输入/输出：
  - 输入：`x`，形状 `[B, C, H, W]`
  - 输出：token，形状 `[B, num_patches, hidden_size]`
- 内部流程：
  1. 先做 patch 卷积
  2. 再做 `1x1` 卷积升维
  3. 把 `NCHW` 排成 `NLC`

## `pMF/models/torch_models.py` 基础层封装

### `TorchLinear`

#### `TorchLinear.__init__`

- 作用：
  模拟原 Flax 风格初始化的线性层封装。
- 输入/输出：
  - 输入：`in_features`, `out_features`, 初始化策略等
  - 输出：模块初始化
- 关键变量或张量形状：
  - 真正的层是 `self._flax_linear`
- 内部流程：
  1. 根据 `weight_init` 选初始化器
  2. 根据 `bias_init` 选偏置初始化器
  3. 创建 `nn.Linear`
  4. 手工初始化参数
- 上游调用者：
  - `RoPEAttention`
  - `FinalLayer`
  - `TimestepEmbedder`
  - `SwiGLUMlp`

#### `TorchLinear.forward`

- 作用：
  调用内部线性层。
- 输入/输出：
  - 输入：任意最后一维匹配 `in_features` 的张量
  - 输出：线性映射结果

### `TorchEmbedding`

#### `TorchEmbedding.__init__`

- 作用：
  封装 embedding 层及其初始化策略。
- 输入/输出：
  - 输入：`num_embeddings`, `embedding_dim`, 初始化参数
  - 输出：模块初始化
- 关键变量或张量形状：
  - `self._flax_embedding`
- 内部流程：
  1. 决定初始化标准差
  2. 创建 `nn.Embedding`
  3. 用正态分布初始化

#### `TorchEmbedding.forward`

- 作用：
  返回 embedding lookup 结果。

### `RMSNorm`

#### `RMSNorm.__init__`

- 作用：
  定义 RMSNorm 层。
- 输入/输出：
  - 输入：`dim`, `eps`
  - 输出：模块初始化

#### `RMSNorm._norm`

- 作用：
  计算按最后一维归一化后的张量。

#### `RMSNorm.forward`

- 作用：
  做 RMS 归一化并乘上可学习缩放。

### `SwiGLUMlp`

#### `SwiGLUMlp.__init__`

- 作用：
  构造 `SwiGLU` 风格的两分支 MLP。
- 输入/输出：
  - 输入：`in_features`, `hidden_features`, 初始化参数
  - 输出：模块初始化
- 关键变量或张量形状：
  - `w1`, `w3` 产生门控与值分支
  - `w2` 投回输入维

#### `SwiGLUMlp.forward`

- 作用：
  计算 `w2(silu(w1(x)) * w3(x))`
- 对应方法含义：
  - 这是现代 transformer 中常见的 gated MLP 实现

## `pMF/meanflow.py` 训练实验主线

这个文件是当前项目最重要的代码中心。它把下面几件事全部塞在一起：

1. 常用工具函数
2. 配置和运行态数据结构
3. U-Net 组件
4. Transformer 组件
5. `meanflow` 与 `pmf` 两套 loss
6. 采样逻辑
7. 数据、模型、优化器工厂
8. checkpoint 与训练循环
9. CLI 入口

阅读它时，建议按下面顺序走：

1. 工具函数和数据结构
2. `Unet` / `MeanFlowTransformer` / `PmfTransformer`
3. `MeanFlowLoss` / `PixelMeanFlowLoss`
4. `generate_*`
5. `build_*`
6. `train`

### 1. 基础工具函数

### `exists`

- 作用：
  判断一个值是否不是 `None`。
- 输入/输出：
  - 输入：任意对象 `x`
  - 输出：布尔值
- 内部流程：
  - 返回 `x is not None`
- 上游调用者：
  - 大量模块初始化和前向逻辑
- 阅读注意点：
  - 这是该文件里最常见的小辅助函数之一

### `default`

- 作用：
  提供“如果为空则回退默认值”的逻辑。
- 输入/输出：
  - 输入：`val`, `d`
  - 输出：`val` 或默认值
- 内部流程：
  1. 如果 `val` 存在，直接返回
  2. 否则如果 `d` 是函数，就调用它
  3. 否则直接返回 `d`
- 上游调用者：
  - `Unet.__init__`
  - 其他构造函数

### `cycle`

- 作用：
  把一个 `DataLoader` 包装成无限迭代器。
- 输入/输出：
  - 输入：`loader`
  - 输出：无限生成 batch 的生成器
- 内部流程：
  - 在 `while True` 中不断遍历 dataloader
- 上游调用者：
  - `train`
- 阅读注意点：
  - 这样训练循环就不用手工处理 epoch 边界

### `configure_runtime`

- 作用：
  统一设置 cudnn 确定性、TF32 和 float32 matmul 精度。
- 输入/输出：
  - 输入：`deterministic`, `matmul_precision`, `allow_tf32`
  - 输出：无
- 内部流程：
  1. 设置 `torch.backends.cudnn.deterministic`
  2. 根据确定性模式切换 `cudnn.benchmark`
  3. 若有 CUDA，设置 matmul / cudnn 的 TF32 开关
  4. 若当前 PyTorch 支持，设置 `torch.set_float32_matmul_precision`
- 阅读注意点：
  - 这是一个“运行时环境开关”辅助函数。
  - 截至当前代码，它还没有接入 `train()` 或 CLI 主流程。

### `set_seed`

- 作用：
  统一设置 Python、NumPy、PyTorch 的随机种子。
- 输入/输出：
  - 输入：`seed`
  - 输出：无
- 内部流程：
  1. 设置 Python `random`
  2. 设置 `numpy`
  3. 设置 `torch`
  4. 若有 CUDA，同步设置所有 CUDA 设备
- 上游调用者：
  - `train`
  - `fid_eval.main`
- 阅读注意点：
  - 训练和 FID 采样都依赖它来保持可复现。
  - cudnn / TF32 这类运行时开关不在这里处理，而是由 `configure_runtime` 负责。

### `to_serializable`

- 作用：
  把配置里的 `Path`、`tuple` 等对象转换成 JSON 可写格式。
- 输入/输出：
  - 输入：任意值
  - 输出：可序列化值
- 内部流程：
  - `Path -> str`
  - `tuple -> list`
  - 其他值原样返回
- 上游调用者：
  - `make_checkpoint`
  - `save_json`

### `channel_last_vector_to_spatial`

- 作用：
  把 `[B, C]` 的条件向量扩成 `[B, C, 1, 1]`，方便加到卷积特征图。
- 输入/输出：
  - 输入：`x`
  - 输出：扩维后的张量
- 上游调用者：
  - `ConvNextBlock.forward`

### `conv_qkv_to_heads`

- 作用：
  把卷积输出的 `q/k/v` 拆成多头格式。
- 输入/输出：
  - 输入：`x: [B, C, H, W]`, `heads`
  - 输出：`[B, heads, dim_head, H*W]`
- 上游调用者：
  - `Attention.forward`
  - `LinearAttention.forward`

### `heads_to_spatial`

- 作用：
  把 attention 输出重新拼回空间特征图。
- 输入/输出：
  - 输入：`[B, heads, tokens, dim_head]`
  - 输出：`[B, heads*dim_head, H, W]`
- 上游调用者：
  - `Attention.forward`

### `heads_channels_to_spatial`

- 作用：
  把线性 attention 的中间表示拼回空间维。
- 输入/输出：
  - 输入：`[B, heads, channels, tokens]`
  - 输出：`[B, heads*channels, H, W]`
- 上游调用者：
  - `LinearAttention.forward`

### 2. 配置与运行态数据结构

### `TrainConfig`

- 作用：
  统一承载训练、采样、模型、优化器、设备等全部配置。
- 输入/输出：
  - 输入：字段初始化值
  - 输出：dataclass 实例
- 上游调用者：
  - `parse_args`
  - `fid_eval.load_run_config`
  - `train`
- 阅读注意点：
  - 这个 dataclass 基本等价于整个 `meanflow.py` 的公共配置接口。
  - `sample_*` 字段主要服务 `generate_pmf()` 和 `fid_eval.py`。
  - `muon_weight_decay` 和 `muon_aux_eps` 用来配置 Muon 主参数组以外的辅助 Adam 组。

### `DatasetSpec`

- 作用：
  存储数据集的通道数、图像尺寸、类别数与归一化参数。
- 上游调用者：
  - `get_dataset_spec`
  - `build_datasets`
  - `save_sample_grid`
  - `fid_eval.generate_fid_samples`

### `RunState`

- 作用：
  存储训练中需要保存/恢复的非参数状态。
- 上游调用者：
  - `train`
  - `make_checkpoint`
  - `restore_checkpoint`

### 3. U-Net 组件

### `Residual`

#### `Residual.__init__`

- 作用：
  用一个简单封装把任意模块变成残差模块。

#### `Residual.forward`

- 作用：
  返回 `fn(x, ...) + x`
- 上游调用者：
  - `Unet`

### `Upsample`

- 作用：
  返回一个 `ConvTranspose2d` 上采样层。
- 输入/输出：
  - 输入：`dim`
  - 输出：`nn.ConvTranspose2d(dim, dim, 4, 2, 1)`
- 上游调用者：
  - `Unet.__init__`

### `Downsample`

- 作用：
  返回一个 `Conv2d` 下采样层。
- 输入/输出：
  - 输入：`dim`
  - 输出：`nn.Conv2d(dim, dim, 4, 2, 1)`
- 上游调用者：
  - `Unet.__init__`

### `SinusoidalPositionEmbeddings`

#### `SinusoidalPositionEmbeddings.__init__`

- 作用：
  记录嵌入维度。

#### `SinusoidalPositionEmbeddings.forward`

- 作用：
  把时间标量编码成正弦位置向量。
- 输入/输出：
  - 输入：`time: [B]`
  - 输出：`[B, dim]`
- 上游调用者：
  - `Unet.time_mlp`
  - `Unet.time_mlp_h`

### `ConvNextBlock`

#### `ConvNextBlock.__init__`

- 作用：
  构造一个带深度卷积、GroupNorm 和时间条件注入的卷积块。
- 输入/输出：
  - 输入：输入/输出通道、时间嵌入维、扩张倍数等
  - 输出：模块初始化
- 关键变量或张量形状：
  - `self.mlp` 把时间向量投成通道条件
  - `self.ds_conv` 是 depthwise conv
  - `self.res_conv` 用于通道数不匹配时的捷径
- 上游调用者：
  - `Unet`

#### `ConvNextBlock.forward`

- 作用：
  处理一次卷积残差块，并可注入时间嵌入。
- 输入/输出：
  - 输入：`x`, `time_emb=None`
  - 输出：更新后的特征图
- 内部流程：
  1. 先做 depthwise conv
  2. 若有时间条件，则投影后广播加到特征图
  3. 走卷积主干
  4. 加残差分支
- 对应方法含义：
  - 这是 U-Net 中“时间条件注入”的主要位置

### `Attention`

#### `Attention.__init__`

- 作用：
  构造标准二维 self-attention。

#### `Attention.forward`

- 作用：
  在特征图上执行全 attention。
- 输入/输出：
  - 输入：`x: [B, C, H, W]`
  - 输出：同形状特征图
- 内部流程：
  1. 用 `1x1 conv` 生成 `q/k/v`
  2. 拆多头
  3. 计算相似度与 softmax
  4. 汇聚 `v`
  5. 拼回空间特征图
- 上游调用者：
  - `Unet.mid_attn`

### `LinearAttention`

#### `LinearAttention.__init__`

- 作用：
  构造线性 attention 版本，降低计算开销。

#### `LinearAttention.forward`

- 作用：
  执行近似线性的 attention 计算。
- 输入/输出：
  - 输入：`x: [B, C, H, W]`
  - 输出：同形状特征图
- 内部流程：
  1. 生成 `q/k/v`
  2. `q` 在通道维 softmax，`k` 在 token 维 softmax
  3. 先汇总 context，再乘回 `q`
  4. 拼回空间特征图
- 上游调用者：
  - `Unet` 的下采样和上采样阶段

### `PreNorm`

#### `PreNorm.__init__`

- 作用：
  用 `GroupNorm` 包一层“先归一化，再调用子模块”的结构。

#### `PreNorm.forward`

- 作用：
  返回 `fn(norm(x))`
- 上游调用者：
  - `Residual(PreNorm(...))`

### `Unet`

#### `Unet.__init__`

- 作用：
  组装完整 U-Net backbone。
- 输入/输出：
  - 输入：基础宽度、输出维、倍率、通道数、时间条件开关等
  - 输出：模块初始化
- 关键变量或张量形状：
  - `downs`: 下采样阶段
  - `mid_block1/mid_attn/mid_block2`: 中间瓶颈
  - `ups`: 上采样阶段
  - `final_conv`: 最终输出层
- 内部流程：
  1. 建初始卷积
  2. 若启用时间条件，建 `time_mlp` 与 `time_mlp_h`
  3. 依层构造 `downs`
  4. 构造中间层
  5. 逆序构造 `ups`
  6. 构造输出头
- 上游调用者：
  - `build_model`

#### `Unet.forward`

- 作用：
  根据当前图像 `x` 和时间条件预测 `u`。
- 输入/输出：
  - 输入：`x`, `time`, `h=None`
  - 输出：`[B, C, H, W]`
- 关键变量或张量形状：
  - `residuals` 保存跳连
- 内部流程：
  1. 初始卷积把输入映到特征空间
  2. 用 `time_mlp(time)` 生成时间嵌入
  3. 如果给了 `h`，再加上 `time_mlp_h(h)`
  4. 逐层下采样：`block1 -> block2 -> attention -> save residual -> downsample`
  5. 经过中间三层
  6. 逐层上采样：拼接跳连后再 `block1 -> block2 -> attention -> upsample`
  7. 经过最终卷积头得到输出
- 上游调用者：
  - `MeanFlowLoss.__call__`
  - `generate_meanflow`
- 下游依赖：
  - 上面所有 U-Net 子模块
- 阅读注意点：
  - `time` 和 `h` 都进入卷积主干，所以 `Unet` 能同时感知当前时间和 two-time 间隔

### 4. Transformer 组件

### `timestep_embedding`

- 作用：
  生成 transformer 侧使用的正余弦时间嵌入。
- 输入/输出：
  - 输入：`timesteps`, `dim`, `max_period`
  - 输出：`[B, dim]`
- 上游调用者：
  - `ScalarConditionEmbed.forward`

### `ScalarConditionEmbed`

#### `ScalarConditionEmbed.__init__`

- 作用：
  为标量条件构造小 MLP。

#### `ScalarConditionEmbed.forward`

- 作用：
  先做 `timestep_embedding`，再过 MLP，得到隐藏向量。
- 上游调用者：
  - `MeanFlowTransformer`
  - `PmfTransformer`

### `LabelConditionEmbed`

#### `LabelConditionEmbed.__init__`

- 作用：
  构造类别 embedding 表。

#### `LabelConditionEmbed.forward`

- 作用：
  返回标签向量。
- 阅读注意点：
  - 这里同样保留了 `num_classes + 1` 个槽位，用于空标签

### `PatchEmbed`

#### `PatchEmbed.__init__`

- 作用：
  把图像按 patch 切块并映射到隐藏维。
- 输入/输出：
  - 输入：图像尺寸、patch 大小、通道数、隐藏维
  - 输出：模块初始化
- 关键变量或张量形状：
  - `num_patches = (image_size // patch_size) ** 2`

#### `PatchEmbed.forward`

- 作用：
  把 `NCHW` 图像变成 `[B, num_patches, hidden_size]`
- 上游调用者：
  - `MeanFlowTransformer.forward`
  - `PmfTransformer.build_sequence`

### `rms_norm_last_dim`

- 作用：
  对最后一维做无参数 RMSNorm。
- 输入/输出：
  - 输入：`x`, `eps=1e-6`
  - 输出：归一化后的张量
- 上游调用者：
  - `FullAttentionResidual.forward`
- 阅读注意点：
  - 这是 residual history 聚合路径里的一个纯函数辅助。

### `FullAttentionResidual`

#### `FullAttentionResidual.__init__`

- 作用：
  定义一个沿深度 history 做聚合的读取器。
- 输入/输出：
  - 输入：`hidden_size`, `eps=1e-6`
  - 输出：模块初始化
- 关键变量或张量形状：
  - `query`: `[hidden_size]` 的可学习伪查询向量

#### `FullAttentionResidual.forward`

- 作用：
  从 `history` 里按深度维做加权汇聚，返回新的 token 表示。
- 输入/输出：
  - 输入：`history`，其中每个元素通常是 `[B, tokens, hidden_size]`
  - 输出：`[B, tokens, hidden_size]`
- 内部流程：
  1. 把 `history` 堆叠成 `[B, tokens, depth, hidden_size]`
  2. 对最后一维做 `rms_norm_last_dim`
  3. 用 `query` 对 depth 维打分
  4. softmax 后对 depth 维加权求和
- 上游调用者：
  - `TransformerBlock.forward`
  - `MeanFlowTransformer.forward`
  - `PmfTransformer.forward`
- 阅读注意点：
  - 当前 residual 模式的关键不在 `TransformerAttention` 内部，而在这条显式 `history` 聚合链。

### `TransformerAttention`

#### `TransformerAttention.__init__`

- 作用：
  构造标准多头 self-attention。
- 输入/输出：
  - 输入：隐藏维、头数、attention 实现名
  - 输出：模块初始化
- 关键变量或张量形状：
  - `qkv`: 线性层输出 `3 * hidden_size`
  - `proj`: 输出投影
- 阅读注意点：
  - `attn_impl` 仍然会被传入，但当前真正的 residual 分支逻辑已经上移到 `TransformerBlock` 和 `FullAttentionResidual`。

#### `TransformerAttention.forward`

- 作用：
  计算一层标准 transformer attention 输出。
- 输入/输出：
  - 输入：`x`
  - 输出：`attn_out`
- 内部流程：
  1. 线性投影并拆成 `q/k/v`
  2. 优先尝试 `scaled_dot_product_attention`
  3. 若当前 kernel / autograd 组合不支持，则退回手工 logits + softmax 实现
  4. 输出投影
- 上游调用者：
  - `TransformerBlock.forward`
- 阅读注意点：
  - `try / except NotImplementedError` 是为了兼容 JVP 等场景。

### `TransformerMlp`

#### `TransformerMlp.__init__`

- 作用：
  定义标准两层 GELU MLP。

#### `TransformerMlp.forward`

- 作用：
  返回 `Linear -> GELU -> Linear` 的结果。

### `TransformerBlock`

#### `TransformerBlock.__init__`

- 作用：
  定义 `LayerNorm + Attention + LayerNorm + MLP` 的 block。
- 阅读注意点：
  - 当 `attn_impl == "residual"` 时，会额外创建 `pre_attn_residual` 和 `pre_mlp_residual` 两个 history 聚合器。

#### `TransformerBlock.forward`

- 作用：
  做一层 transformer 更新；在 residual 模式下，消费并扩展 `history`。
- 输入/输出：
  - 输入：`x: torch.Tensor | None = None`, `history: list[torch.Tensor] | None = None`
  - 输出：`(block_out, history_or_none)`
- 内部流程：
  1. residual 模式下，先用 `pre_attn_residual(history)` 聚合出 `attn_in`
  2. 做 `attn(norm1(attn_in))`，并把 `attn_out` 追加进 `history`
  3. 再用 `pre_mlp_residual(history)` 聚合出 `mlp_in`
  4. 做 `mlp(norm2(mlp_in))`
  5. residual 模式返回“本层新增的 `mlp_out` + 更新后的 history”
  6. naive 模式才走普通 `x + attn + mlp` 残差更新
- 上游调用者：
  - `MeanFlowTransformer`
  - `PmfTransformer`
- 阅读注意点：
  - residual 模式下 `x` 不是必需输入，真正的主输入变成了 `history`。
  - 返回值里的第一个张量不再代表“累计后的整条 hidden state”，而是“当前 block 新产生的输出”。

### `MeanFlowTransformer`

#### `MeanFlowTransformer.__init__`

- 作用：
  组装 `meanflow` 变体使用的 transformer 骨干。
- 输入/输出：
  - 输入：图像尺寸、patch、大模型宽深、头数、MLP 比例、attention 实现
  - 输出：模块初始化
- 关键变量或张量形状：
  - `patch_embed`
  - `time_embed`
  - `delta_embed`
  - `pos_embed`
  - `blocks`
  - `output_residual`
  - `head`
- 上游调用者：
  - `build_model`

#### `MeanFlowTransformer.unpatchify`

- 作用：
  把 patch token 恢复成整图。

#### `MeanFlowTransformer.forward`

- 作用：
  在 transformer 骨干上预测 `u`。
- 输入/输出：
  - 输入：`x`, `time`, `h=None`
  - 输出：图像形状的 `u`
- 内部流程：
  1. `patch_embed(x)` 得到图像 token
  2. `time_embed(time)` 生成条件
  3. 若有 `h`，再加上 `delta_embed(h)`
  4. 对每个 patch token 加位置编码和条件向量
  5. naive 模式下直接串行通过所有 block
  6. residual 模式下先维护 `history = [tokens]`，再让每个 block 只负责扩展 history
  7. residual 模式结束后，用 `output_residual(history)` 汇聚出最终 token
  8. `norm + head`
  9. `unpatchify`
- 上游调用者：
  - `MeanFlowLoss.__call__`
  - `generate_meanflow`
- 对应方法含义：
  - 与 U-Net 版 `meanflow` 相比，条件注入方式从卷积块改成了 token 加法。
  - residual 模式下最终输出不是“最后一层 token 原样透传”，而是“对整条 depth history 再做一次聚合”。

### `PmfTransformer`

#### `PmfTransformer.__init__`

- 作用：
  组装 `pmf` 训练版本的 transformer。
- 输入/输出：
  - 输入：图像/patch/隐藏维/层数/头数/类别数/attention 实现/噪声缩放
  - 输出：模块初始化
- 关键变量或张量形状：
  - `null_label = num_classes`
  - 5 个前缀 token：`label/omega/t_min/t_max/h`
  - `shared_blocks`, `u_heads`, `v_heads`
  - `u_output_residual`, `v_output_residual`
  - `u_head`, `v_head`
- 内部流程：
  1. 检查深度至少为 3
  2. 创建 patch embed 和各种条件 embedder
  3. 创建 5 个 learnable prefix token
  4. 创建位置编码
  5. 拆出共享层和双头层
  6. residual 模式下额外创建 `u/v` 两个输出聚合器
  7. 初始化规范化层和输出头
- 上游调用者：
  - `build_model`

#### `PmfTransformer.unpatchify`

- 作用：
  把 patch 输出恢复成图像。

#### `PmfTransformer.build_sequence`

- 作用：
  构造 `[label, omega, t_min, t_max, h, image_tokens]` 序列。
- 输入/输出：
  - 输入：`x, h, omega, t_min, t_max, labels`
  - 输出：完整 token 序列
- 内部流程：
  1. embed 图像 patch
  2. embed 每个条件并加到各自 token 模板上
  3. 拼接前缀与图像 token
  4. 加位置编码
- 阅读注意点：
  - `omega` 先被转换成 `1 - 1 / omega`

#### `PmfTransformer.forward`

- 作用：
  输出 `u` 和 `v` 两个速度场。
- 输入/输出：
  - 输入：`x, time, h, omega, t_min, t_max, labels`
  - 输出：`(u, v)`
- 关键变量或张量形状：
  - `tokens` 先过共享主干，再分叉
  - 最终输出 `u_pixels/v_pixels` 会再转成速度量
- 内部流程：
  1. `build_sequence`
  2. naive 模式下：共享 block 串行更新 `tokens`，再分别走 `u_heads` 与 `v_heads`
  3. residual 模式下：先建立 `shared_history = [tokens]`，共享主干只负责扩展这条 history
  4. 基于同一份共享 history 复制出 `u_history` 和 `v_history`
  5. `u_heads` / `v_heads` 分别继续扩展各自 history
  6. 用 `u_output_residual(u_history)` 和 `v_output_residual(v_history)` 取得最终 token
  7. 去掉前缀 token，只保留图像 token
  8. `u_norm/v_norm + u_head/v_head + unpatchify`
  9. 用 `(x - pred_pixels) / clamp(time)` 转成 `u/v`
- 上游调用者：
  - `PixelMeanFlowLoss`
  - `generate_pmf`
- 阅读注意点：
  - 和 `pmfDiT` 一样，这里真正进入 token 条件的是 `h` 等量，而不是 `time`。
  - `time` 在这里主要用于最后从像素预测反推速度场。
- 对应方法含义：
  - 这是当前训练主线里 pMF 的核心模型实现

### 5. 扰动构造与时间采样

### `build_meanflow_corruption`

- 作用：
  构造线性插值扰动 `z_t = (1-t)x0 + t eps`
- 输入/输出：
  - 输入：`x0`, `eps`, `tau`
  - 输出：`z_t`
- 上游调用者：
  - `MeanFlowLoss.__call__`
  - `PixelMeanFlowLoss.__call__`

### `sample_time_from_noise`

- 作用：
  根据指定分布采样时间。
- 输入/输出：
  - 输入：形状、设备、分布名、`p_mean/p_std/t_min`
  - 输出：时间张量
- 上游调用者：
  - 当前主流程中没有直接使用
- 阅读注意点：
  - 这是一个保留下来的通用辅助函数，实际 loss 内部有各自的采样实现

### 6. `MeanFlowLoss`

### `MeanFlowLoss.__init__`

- 作用：
  保存时间采样分布与自适应加权参数。

### `MeanFlowLoss.noise_distribution`

- 作用：
  按 `logit_normal` 或 `uniform` 采样原始时间噪声。
- 输入/输出：
  - 输出值仍在 `[0, 1]` 范围

### `MeanFlowLoss.__call__`

- 作用：
  计算 `meanflow` 变体的训练损失。
- 输入/输出：
  - 输入：`net`, `images`, `labels=None`
  - 输出：标量 loss
- 关键变量或张量形状：
  - `tau`: `[B,1,1,1]`
  - `r`: 全零
  - `v_g`: 扰动轨迹对时间的一阶导
  - `u_target`: 用 JVP 推出来的目标平均速度
- 内部流程：
  1. 采样时间 `tau`
  2. 采样噪声 `eps`
  3. 构造扰动函数 `corruption_of_tau`
  4. 用 `torch.func.jvp` 同时得到 `x_t` 和轨迹导数 `v_g`
  5. 定义 `u_wrapper`，把 `net` 包成只输出 `u` 的函数
  6. 再做一次 JVP，求 `u` 及其沿轨迹方向的导数 `du_dt`
  7. 用 `u_target = v_g - h * du_dt` 构造监督目标
  8. 计算误差平方
  9. 用误差反比权重做自适应加权
  10. 返回 batch 均值
- 上游调用者：
  - `train`
  - `evaluate_loss`
- 下游依赖：
  - `build_meanflow_corruption`
  - `torch.func.jvp`
- 对应方法含义：
  - 这里是 MeanFlow identity 落地成训练目标的地方

### 7. `PixelMeanFlowLoss`

### `PixelMeanFlowLoss.__init__`

- 作用：
  保存 pMF 训练所需的所有分布参数、CFG 参数与自适应加权参数。

### `PixelMeanFlowLoss.noise_distribution`

- 作用：
  采样 `[0, 1]` 上的时间变量。

### `PixelMeanFlowLoss.sample_tr`

- 作用：
  采样 `(t, r)` 并区分哪些样本走 flow matching 特例。
- 输入/输出：
  - 输出：`t`, `r`, `fm_mask`
- 内部流程：
  1. 分别采样 `t` 和 `r`
  2. 如果开启 `tr_uniform`，给一部分样本改成均匀采样
  3. 按 `data_proportion` 决定多少样本强制 `r=t`
  4. 排序成 `t >= r`
- 对应方法含义：
  - `fm_mask=True` 的样本退化到 flow matching 情况

### `PixelMeanFlowLoss.sample_cfg_scale`

- 作用：
  采样 CFG 强度 `omega`。
- 输入/输出：
  - 输出：`[B,1,1,1]`
- 内部流程：
  - 当 `cfg_beta == 1.0` 时使用指数形式，否则走更一般的 beta 形式

### `PixelMeanFlowLoss.sample_cfg_interval`

- 作用：
  为每个样本采样 CFG 生效区间 `[t_min, t_max]`。
- 输入/输出：
  - 输入：`batch`, `device`, `fm_mask`
  - 输出：`t_min`, `t_max`
- 内部流程：
  1. 一般样本随机采区间
  2. `fm_mask=True` 的样本固定成 `[0,1]`

### `PixelMeanFlowLoss.v_cond_fn`

- 作用：
  调用模型拿到带条件的 `v` 头输出。
- 输入/输出：
  - 输入：`net, z_t, t, omega, labels`
  - 输出：`v`
- 阅读注意点：
  - 这里把 `h` 固定成零，把区间固定成 `[0,1]`

### `PixelMeanFlowLoss.v_fn`

- 作用：
  一次前向同时得到条件版 `v_c` 和无条件版 `v_u`。
- 输入/输出：
  - 输出：`(v_c, v_u)`
- 内部流程：
  1. 把 batch 复制一份拼成双倍 batch
  2. 一半用真实标签，一半用 `null_label`
  3. 条件强度一半用 `omega`，一半用 `1`
  4. 再拆回条件/无条件结果
- 对应方法含义：
  - 这是 classifier-free guidance 训练侧的批内并行技巧

### `PixelMeanFlowLoss.guidance_fn`

- 作用：
  构造目标指导速度 `v_g`。
- 输入/输出：
  - 输入：模型、当前 `v_t/z_t/t/r/labels/fm_mask/omega/t_min/t_max`
  - 输出：`(v_g, v_c_interval)`
- 内部流程：
  1. 求 `v_c` 和 `v_u`
  2. 对 `fm_mask` 样本使用全程 CFG 公式
  3. 对其余样本只在区间内启用 CFG
  4. 输出目标指导速度和区间条件版 `v`
- 阅读注意点：
  - 这里 `r` 没实际用到，因此函数开头显式 `del r`

### `PixelMeanFlowLoss.cond_drop`

- 作用：
  按概率做类别条件 dropout。
- 输入/输出：
  - 输出：新的 `labels` 与新的 `v_g`
- 内部流程：
  1. 随机采 `drop_mask`
  2. 被丢弃的样本把标签改成 `null_label`
  3. 对应目标速度退回 `v_t`

### `PixelMeanFlowLoss.adaptive_weight`

- 作用：
  做自适应加权，降低大损失样本的直接主导性。
- 输入/输出：
  - 输入：逐样本 loss
  - 输出：重加权后的逐样本 loss

### `PixelMeanFlowLoss.__call__`

- 作用：
  计算 pMF 训练损失。
- 输入/输出：
  - 输入：`net`, `images`, `labels`
  - 输出：标量 loss
- 关键变量或张量形状：
  - `t/r/fm_mask`
  - `z_t`, `v_t`
  - `omega`, `t_min`, `t_max`
  - `v_g`, `v_c`
  - `(u, v), (du_dt, _)`
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
- 上游调用者：
  - `train`
  - `evaluate_loss`
- 下游依赖：
  - `sample_tr`
  - `guidance_fn`
  - `cond_drop`
  - `torch.func.jvp`
- 对应方法含义：
  - 这里把 pMF 的“图像式预测 + 速度空间监督 + CFG 训练”全部串起来了

### 8. 采样函数

### `generate_meanflow`

- 作用：
  生成 `meanflow` 变体样本。
- 输入/输出：
  - 输入：`model`, `noise_x`
  - 输出：生成样本
- 内部流程：
  1. 把 `tau` 和 `h` 固定成全 1
  2. 直接返回 `noise_x - model(noise_x, tau, h)`
- 阅读注意点：
  - 这是单步生成

### `build_sample_labels`

- 作用：
  为 pMF 采样构造循环标签。
- 输入/输出：
  - 输入：类别数、batch、device
  - 输出：`[0,1,2,...] % num_classes`
- 上游调用者：
  - `train`

### `generate_pmf`

- 作用：
  用离散时间表执行 pMF 采样。
- 输入/输出：
  - 输入：模型、噪声、标签、步数、采样 CFG 参数
  - 输出：采样后的图像
- 内部流程：
  1. 初始噪声乘上 `model.noise_scale`
  2. 构造 `1 -> 0` 的时间表
  3. 每一步：
     1. 扩展 `t/r`
     2. 构造固定 `omega/t_min/t_max`
     3. 调模型拿 `u`
     4. 更新 `z_t`
- 上游调用者：
  - `generate_samples`

### `generate_samples`

- 作用：
  统一包装两种变体的采样入口。
- 输入/输出：
  - 输入：`model`, `noise_x`, `variant`, `labels=None`, `config=None`
  - 输出：采样结果
- 内部流程：
  - `variant == "meanflow"` 时走 `generate_meanflow`
  - `variant == "pmf"` 时走 `generate_pmf`
- 上游调用者：
  - `train`
  - `fid_eval.generate_fid_samples`

### 9. 数据、模型、优化器、loss 工厂

### `get_dataset_spec`

- 作用：
  返回 `mnist` 或 `cifar10` 的标准规格。
- 输入/输出：
  - 输入：数据集名
  - 输出：`DatasetSpec`

### `build_datasets`

- 作用：
  构造训练集、验证集和对应数据规格。
- 输入/输出：
  - 输入：`config`
  - 输出：`train_dataset, eval_dataset, spec`
- 内部流程：
  - 根据数据集选择不同的 transform 和 torchvision 数据集类

### `build_dataloader`

- 作用：
  为给定数据集创建 `DataLoader`。
- 输入/输出：
  - 输入：dataset、batch_size、shuffle、worker 数、seed、drop_last
  - 输出：`DataLoader`
- 关键变量或张量形状：
  - 内部创建了带固定种子的 `torch.Generator`

### `build_model`

- 作用：
  根据 `variant/backbone` 选择模型。
- 输入/输出：
  - 输入：`config`, `spec`
  - 输出：`nn.Module`
- 内部流程：
  1. `variant == "pmf"` 时强制要求 `backbone == "transformer"`，返回 `PmfTransformer`
  2. 否则如果 `backbone == "unet"`，返回 `Unet`
  3. 如果 `backbone == "transformer"`，返回 `MeanFlowTransformer`
- 上游调用者：
  - `train`
  - `fid_eval.main`
- 阅读注意点：
  - `pmf` 在当前实验链中只支持 transformer

### `build_optimizer`

- 作用：
  按配置构造 `Adam`、`AdamW` 或 `Muon`。
- 输入/输出：
  - 输入：`model`, `config`
  - 输出：优化器实例
- 内部流程：
  1. 若选 `adam` / `adamw`，直接返回标准优化器
  2. 若选 `muon`：
     1. 检查只用于 `variant=pmf`
     2. 检查模型必须是 `PmfTransformer`
     3. 按参数名和张量维度拆成 `muon_params` 与 `adamw_params`
     4. 构建 param groups
     5. 返回 `SingleDeviceMuonWithAuxAdam`
- 阅读注意点：
  - 只有 shared/u/v block 中的矩阵参数会走 Muon

### `build_loss`

- 作用：
  根据变体返回正确 loss 对象。
- 输入/输出：
  - 输入：`config`
  - 输出：`MeanFlowLoss` 或 `PixelMeanFlowLoss`

### 10. 运行目录、样本保存与 checkpoint

### `get_run_dir`

- 作用：
  决定当前训练 run 的目录。
- 输入/输出：
  - 输入：`config`
  - 输出：`Path`
- 内部流程：
  - 如果 `resume` 非空，则回到 checkpoint 的上上级目录
  - 否则按时间戳创建新目录

### `save_json`

- 作用：
  以 UTF-8 写 JSON 文件，并自动创建父目录。

### `save_sample_grid`

- 作用：
  把生成样本反归一化后存成网格图。
- 输入/输出：
  - 输入：`samples`, `spec`, `path`
  - 输出：无
- 内部流程：
  1. 用 `spec.mean/std` 反归一化
  2. `make_grid`
  3. `save_image`

### `evaluate_loss`

- 作用：
  用少量 batch 估计验证损失。
- 输入/输出：
  - 输入：模型、loss、loader、device、`num_batches`
  - 输出：平均 loss
- 内部流程：
  1. 切到 `eval`
  2. 取若干 batch
  3. 逐批计算 loss
  4. 恢复 `train`
- 上游调用者：
  - `train`

### `make_checkpoint`

- 作用：
  打包模型、优化器、配置、训练状态与 RNG 状态。
- 输入/输出：
  - 输入：`model`, `optimizer`, `config`, `state`
  - 输出：checkpoint dict
- 关键变量或张量形状：
  - RNG 状态包括 `torch/numpy/python/cuda`
- 上游调用者：
  - `train`

### `_coerce_rng_state_tensor`

- 作用：
  把不同格式的 RNG 状态统一转成 CPU 上的 `ByteTensor`。

### `_restore_rng_state`

- 作用：
  从 checkpoint 恢复随机数状态。
- 输入/输出：
  - 输入：`rng_state`
  - 输出：无
- 上游调用者：
  - `restore_checkpoint`

### `restore_checkpoint`

- 作用：
  从 checkpoint 恢复模型、优化器和运行态。
- 输入/输出：
  - 输入：路径、模型、优化器、设备
  - 输出：`RunState`
- 内部流程：
  1. `torch.load`
  2. `model.load_state_dict`
  3. 若 pMF 旧 checkpoint 不兼容，则抛出带解释的错误
  4. 恢复优化器
  5. 恢复 RNG
  6. 重建 `RunState`
- 上游调用者：
  - `train`
  - `fid_eval.main`

### 11. CLI 与训练循环

### `parse_args`

- 作用：
  解析命令行参数并转成 `TrainConfig`。
- 输入/输出：
  - 输入：CLI
  - 输出：`TrainConfig`
- 阅读注意点：
  - CLI 字段和 `TrainConfig` 字段基本一一对应
  - `device` 若为空会回退到 `TrainConfig().device`

### `train`

- 作用：
  执行一次完整训练并返回 run 目录。
- 输入/输出：
  - 输入：`config`
  - 输出：`run_dir`
- 关键变量或张量形状：
  - `train_iter` 是无限 batch 生成器
  - `fixed_noise` / `fixed_labels` 用于周期性采样可视化
- 内部流程：
  1. 检查 `grad_accum_steps >= 1`
  2. 设随机种子
  3. 创建 run/sample/checkpoint 目录
  4. 构造训练与验证数据
  5. 构造模型、优化器、loss
  6. 如果 `resume`，恢复 checkpoint
  7. 保存 `config.json`
  8. 构造固定采样噪声和固定标签
  9. 进入主训练循环
  10. 每个 step：
      1. `zero_grad`
      2. 按 `grad_accum_steps` 取 microbatch
      3. 计算 loss、反传并累计
      4. clip grad
      5. `optimizer.step`
      6. 更新 `RunState`
      7. 按间隔决定是否验证、采样、保存
  11. 保存最新/best checkpoint 与 `metrics.json`
  12. 返回 `run_dir`
- 上游调用者：
  - `main`
- 下游依赖：
  - 几乎所有 `build_*`、`generate_samples`、`save_*`、checkpoint 函数
- 阅读注意点：
  - 这是整份文件的控制中心
  - `state.best_val_loss` 既用于进度条展示，也用于决定何时写 `best.pt`

### `main`

- 作用：
  命令行入口，解析参数并启动训练。
- 输入/输出：
  - 输入：无
  - 输出：无
- 内部流程：
  1. `config = parse_args()`
  2. `run_dir = train(config)`
  3. 打印运行目录

## `pMF/optim/muon.py` Muon 优化器实现

这个文件负责把“部分参数走 Muon、部分参数走 Adam 风格更新”的逻辑落地。

### `zeropower_via_newtonschulz5`

- 作用：
  用 Newton-Schulz 迭代近似梯度矩阵的零次幂归一化结果。
- 输入/输出：
  - 输入：`grad`, `steps`
  - 输出：归一化后的 update 张量
- 关键变量或张量形状：
  - 如果梯度是 4D，会在上游先 reshape 成矩阵
  - 内部在 `bfloat16` 上运算
- 内部流程：
  1. 检查输入至少是二维
  2. 若行数大于列数，先转置
  3. 先做范数归一化
  4. 进行若干次 Newton-Schulz 迭代
  5. 必要时再转回原布局
- 上游调用者：
  - `muon_update`

### `muon_update`

- 作用：
  对单个参数的梯度执行 Muon 更新规则。
- 输入/输出：
  - 输入：`grad`, `momentum`, `beta`, `ns_steps`, `nesterov`
  - 输出：update
- 内部流程：
  1. 更新标准 momentum buffer
  2. 若开 Nesterov，则用 `grad + beta * momentum`
  3. 若梯度是卷积核，拉平成矩阵
  4. 调 `zeropower_via_newtonschulz5`
  5. 按矩阵长宽比做额外缩放
- 上游调用者：
  - `SingleDeviceMuonWithAuxAdam.step`
- 阅读注意点：
  - 这里显式注明 Muon 用的是标准 momentum buffer，而不是 EMA 风格

### `adam_update`

- 作用：
  对单个参数执行 Adam 风格更新。
- 输入/输出：
  - 输入：`grad`, `exp_avg`, `exp_avg_sq`, `step`, `betas`, `eps`
  - 输出：update
- 内部流程：
  1. 更新一阶矩和二阶矩
  2. 做 bias correction
  3. 计算 `exp_avg / (sqrt(exp_avg_sq) + eps)`
- 上游调用者：
  - `SingleDeviceMuonWithAuxAdam.step`

### `SingleDeviceMuonWithAuxAdam`

#### `SingleDeviceMuonWithAuxAdam.__init__`

- 作用：
  规范化参数组，允许同一个优化器里同时管理 Muon 组和 Adam 组。
- 输入/输出：
  - 输入：`param_groups`
  - 输出：优化器实例
- 内部流程：
  1. 检查每个 group 都声明了 `use_muon`
  2. 对 Muon 组补齐 `lr/momentum/ns_steps/nesterov`
  3. 对 Adam 组补齐 `lr/betas/eps`
  4. 调用父类构造器
- 上游调用者：
  - `build_optimizer`

#### `SingleDeviceMuonWithAuxAdam.step`

- 作用：
  对所有参数组执行一步更新。
- 输入/输出：
  - 输入：可选 `closure`
  - 输出：可选 loss
- 内部流程：
  1. 若提供 `closure` 则先重新计算 loss
  2. 遍历参数组
  3. 对 Muon 组：
     1. 懒初始化 `momentum_buffer`
     2. 用 `muon_update` 求更新量
     3. 先做 weight decay
     4. 再更新参数
  4. 对 Adam 组：
     1. 懒初始化 `exp_avg/exp_avg_sq/step`
     2. 用 `adam_update` 求更新量
     3. 先做 weight decay
     4. 再更新参数
- 上游调用者：
  - `train`

## `pMF/utils/torch_util.py` 设备与随机数辅助

### `seed`

- 作用：
  设置 PyTorch 随机种子。
- 输入/输出：
  - 输入：`seed`
  - 输出：无
- 上游调用者：
  - `evaluate.main`

### `device_get`

- 作用：
  把 `torch.Tensor` 搬到 CPU 并转成 `numpy.float32`
- 输入/输出：
  - 输入：tensor
  - 输出：numpy 数组
- 上游调用者：
  - `evaluate.run_evaluate`
  - `evaluate.main`

### `device_put`

- 作用：
  把 numpy 或 tensor 搬到当前分布式设备。
- 输入/输出：
  - 输入：数组或 tensor
  - 输出：位于 `dist.local_device()` 的对象
- 上游调用者：
  - `evaluate.main`

### `BatchGenerator`

#### `BatchGenerator.__init__`

- 作用：
  为 batch 中每个样本维护一个独立 `torch.Generator`。
- 输入/输出：
  - 输入：设备、seed 列表
  - 输出：对象初始化
- 阅读注意点：
  - 这是为了复现类似 JAX/EDM 的“每样本独立随机源”语义

#### `BatchGenerator.randn`

- 作用：
  为 batch 中每个 seed 生成独立高斯噪声。
- 输入/输出：
  - 输入：`size`, 以及传给 `torch.randn` 的额外参数
  - 输出：按 batch 堆叠后的噪声张量
- 内部流程：
  1. 检查 `size[0]` 与 generator 数量一致
  2. 对每个 generator 分别采样
  3. 把结果堆叠并搬到目标设备

#### `BatchGenerator.randn_like`

- 作用：
  生成与给定张量同形状的独立噪声。
- 输入/输出：
  - 输入：`input`
  - 输出：与 `input` 同 shape/dtype/device 风格的噪声
- 内部流程：
  - 复用 `randn`

#### `BatchGenerator.randint`

- 作用：
  为 batch 中每个 seed 独立采样整数。
- 输入/输出：
  - 输入：`torch.randint` 所需参数与 `size`
  - 输出：堆叠后的整数张量
- 内部流程：
  1. 检查 batch 维一致
  2. 对每个 generator 独立采样
  3. 堆叠并搬到目标设备

### `tree_map`

- 作用：
  递归地把函数 `f` 应用到嵌套结构中的叶子节点。
- 输入/输出：
  - 支持 `dict/list/tuple`
- 上游调用者：
  - 当前主流程中未直接使用，但作为通用工具保留

## `pMF/utils/torch_dist_util.py` 分布式辅助

这个文件主要服务 `evaluate.py` 的分布式采样/FID 流程。

### `all_reduce`

- 作用：
  薄封装 `torch.distributed.all_reduce`

### `initialize`

- 作用：
  初始化分布式环境。
- 输入/输出：
  - 输出：无
- 内部流程：
  1. 若环境变量不存在则补默认值
  2. 选择 backend
  3. 初始化进程组
  4. 设置当前 CUDA 设备
  5. 初始化多进程统计辅助状态
- 上游调用者：
  - `evaluate.main`
- 阅读注意点：
  - 单机单卡也会走这套初始化逻辑

### `process_index`

- 作用：
  返回当前 rank。

### `process_count`

- 作用：
  返回 world size。

### `local_device`

- 作用：
  返回当前进程应使用的 CUDA 设备。

### `should_stop`

- 作用：
  当前总是返回 `False`，保留了一个可扩展的停止钩子。

### `update_progress`

- 作用：
  当前为空操作，作为接口占位。

### `print0`

- 作用：
  只在 rank 0 打印日志。

### `_all_gather_tensor`

- 作用：
  从所有进程收集同形状 tensor。

### `all_gather`

- 作用：
  支持对单 tensor 或 tensor 字典做 all-gather。

### `barrier`

- 作用：
  若分布式已初始化，则执行同步屏障。

### `init_multiprocessing`

- 作用：
  初始化文件内部用于跨进程统计的全局状态。
- 输入/输出：
  - 输入：`rank`, `sync_device`
  - 输出：无
- 阅读注意点：
  - 这部分来自 EDM 的统计工具改写版，目前只保留了初始化最小逻辑

## `pMF/utils/fidelity_wrapper.py` FID 封装

这个文件的重点不是重新实现 `torch-fidelity`，而是补上它原生不支持的“URL 指向 `.npz` 参考统计文件”能力。

### `url_to_path`

- 作用：
  如果输入是 URL，就把参考统计文件下载到本地缓存；如果本来就是路径，则原样返回。
- 输入/输出：
  - 输入：`url_or_path`
  - 输出：本地路径
- 内部流程：
  1. 判断是否以 `http://` 或 `https://` 开头
  2. 若是 URL，则下载到临时缓存目录
  3. 返回缓存路径
- 上游调用者：
  - `calculate_metrics`

### `calculate_metrics`

- 作用：
  在保留 `torch-fidelity` 主要逻辑的同时，支持从 `.npz` 参考统计文件计算 FID。
- 输入/输出：
  - 输入：一组 `torch-fidelity` 风格 kwargs
  - 输出：指标字典
- 内部流程：
  1. 先补默认 feature extractor 与 feature layer
  2. 解析是否要算 `isc/fid/kid/ppl`
  3. 对输入 1 提取特征
  4. 若算 ISC，则直接由特征得分
  5. 若算 FID：
     1. 先求输入 1 的统计量
     2. 用 `url_to_path` 解析输入 2
     3. 从 `.npz` 里读 `mu/sigma`
     4. 交给 `fid_statistics_to_metric`
  6. 若算 PPL，则额外调用 `calculate_ppl`
- 上游调用者：
  - `evaluate.run_evaluate`
  - `fid_eval.main`
- 阅读注意点：
  - 这个函数的长 docstring 大多来自上游库；真正与本仓库强相关的改动是“支持 URL/.npz 参考统计”

## 评估、运行与可视化脚本

### `pMF/run.md`

- 作用：
  汇总当前 `meanflow.py` 训练主线的实验命令模板、恢复训练命令和 FID 命令。
- 阅读注意点：
  - 它更像“实验操作手册”，不是运行时代码。
  - 当你已经读懂 `TrainConfig` 字段后，再看这个文件会非常顺。

### `pMF/scripts/run_cifar10_suite.sh`

- 作用：
  批量顺序启动多组 CIFAR10 实验，并把控制台日志整理到统一目录。
- 输入/输出：
  - 输入：脚本内部写死的多组 `python meanflow.py ...` 参数
  - 输出：`runs/cifar10_suite_logs/<timestamp>/suite.log` 与各实验日志
- 内部流程：
  1. 切到 `pMF/` 根目录
  2. 设置 `PYTORCH_CUDA_ALLOC_CONF`
  3. 为本轮 suite 创建时间戳日志目录
  4. 通过 `run_experiment()` 依次执行多组 `meanflow.py`
  5. 用 `tee` 同时写总日志和单实验日志
- 阅读注意点：
  - 这是“批量调度层”，不会改变 `meanflow.py` 的训练逻辑。
  - 当前脚本覆盖了 `meanflow` 基线、`pMF + AdamW + naive`、`pMF + Muon + naive`、`pMF + Muon + residual`。

### `pMF/scripts/plot_cifar_metrics.py`

- 作用：
  读取三组 CIFAR10 run 的 `metrics.json/config.json`，绘制训练损失与验证损失对比图。
- 输入/输出：
  - 输入：脚本顶部 `RUNS` 常量里给定的 run 目录
  - 输出：`assets/showcase/cifar_metrics_comparison.png`
- 内部流程：
  1. 逐个读取 run 的 `metrics.json` 与 `config.json`
  2. 对训练损失做 EMA 平滑
  3. 绘制 train / val 两个子图
  4. 标记最优验证点
  5. 保存到 `assets/showcase/`

### `pMF/scripts/plot_fixcheck_metrics.py`

- 作用：
  针对 fixcheck 这一批 run 生成对比图，重点看修正后的 pMF 实验曲线。
- 输入/输出：
  - 输入：脚本顶部 `RUNS` 常量里给定的 fixcheck run 目录
  - 输出：`assets/showcase/fixcheck_metrics_comparison.png`
- 内部流程：
  1. 读取每个 run 的 `metrics.json` 与 `config.json`
  2. 对 train loss 做 EMA 平滑
  3. 绘制验证曲线并标出最优点
  4. 在图下注记 best step / best val loss
  5. 保存到 `assets/showcase/`
- 阅读注意点：
  - 这个脚本和 `plot_cifar_metrics.py` 的结构几乎一致，只是 run 列表和图标题换成了 fixcheck 版本。

### `pMF/evaluate.py`

这个脚本面向独立推理链：`pixelMeanFlow + pmfDiT`。

### `print0`

- 作用：
  只在 rank 0 打印。
- 上游调用者：
  - `run_evaluate`
  - `main`

### `run_evaluate`

- 作用：
  分布式生成图像、写盘，并调用 FID 计算。
- 输入/输出：
  - 输入：模型、输出目录、FID 参考、样本数、单卡 batch、seed 等
  - 输出：`(fid, inception_score)`
- 内部流程：
  1. 计算总共需要多少 generation step
  2. 创建 `fid_outputs/`
  3. 构造按类别均衡的标签序列
  4. 每个 step：
     1. 计算本 rank 负责的样本索引区间
     2. 用 `BatchGenerator` 和 `model.generate` 采样
     3. 反归一化后逐张写 PNG
  5. 所有进程同步
  6. rank 0 用 `calculate_metrics` 求 FID/IS
  7. 需要时删除生成图片目录
- 阅读注意点：
  - 这是老式“先落图，再从目录算 FID”的流程

### `get_args_parser`

- 作用：
  定义 `sample` / `evaluate` 两种模式的 CLI 参数。
- 输入/输出：
  - 输出：`ArgumentParser`
- 关键参数：
  - `mode`
  - `workdir`
  - `ckpt-path`
  - `model`
  - `num-sampling-steps`
  - `cfg-omega`
  - `interval-min`
  - `interval-max`
  - `num-images`
  - `gen-bsz`
  - `save-samples`
  - `fid-ref`

### `main`

- 作用：
  初始化分布式环境，加载 `pixelMeanFlow`，然后按模式执行采样或评估。
- 输入/输出：
  - 输入：解析后的 args
  - 输出：无
- 内部流程：
  1. `dist.initialize()`
  2. 创建工作目录
  3. 设随机种子
  4. 根据模型名创建 `pixelMeanFlow`
  5. 加载 checkpoint
  6. 把模型搬到设备
  7. 若 `mode == "evaluate"`，调用 `run_evaluate`
  8. 若 `mode == "sample"`，采一批固定标签样本并保存拼图

### `pMF/fid_eval.py`

这个脚本面向 `meanflow.py` 训练产物：它读取 `run_dir/config.json` 和 checkpoint，直接调用当前实验主线中的 `build_model`、`restore_checkpoint`、`generate_samples`。

### `parse_args`

- 作用：
  定义 FID 脚本的 CLI 参数。
- 关键参数：
  - `--run-dir`
  - `--ckpt-path`
  - `--checkpoint`
  - `--data-root`
  - `--ref`
  - `--num-samples`
  - `--gen-bsz`
  - `--fid-batch-size`
  - `--sample-seed`
  - `--sample-steps`
  - `--sample-omega`
  - `--sample-t-min`
  - `--sample-t-max`
  - `--device`
  - `--datasets-download`
  - `--keep-samples`
  - `--isc`

### `load_run_config`

- 作用：
  从 `run_dir/config.json` 里恢复 `TrainConfig`。
- 输入/输出：
  - 输入：`run_dir`
  - 输出：`TrainConfig`
- 内部流程：
  1. 读取 JSON
  2. 拿默认 `TrainConfig` 作为基线
  3. 只合并合法字段
  4. 对 `dim_mults` 做 `list -> tuple`
- 上游调用者：
  - `main`

### `resolve_device`

- 作用：
  把 `"auto" | "cpu" | "cuda"` 映射成 `torch.device`，并在请求 CUDA 但不可用时抛错。

### `resolve_checkpoint`

- 作用：
  解析最终使用的 checkpoint 路径。
- 输入/输出：
  - 若用户显式给 `--ckpt-path`，优先使用
  - 否则默认找 `run_dir/checkpoints/{best|last}.pt`

### `denormalize_images`

- 作用：
  把训练空间的样本张量反归一化到 `[0,1]`
- 上游调用者：
  - `generate_fid_samples`

### `generate_fid_samples`

- 作用：
  按批生成样本并写成 PNG，供 FID 计算使用。
- 输入/输出：
  - 输入：模型、config、输出目录、设备、样本数、batch、seed
  - 输出：无
- 内部流程：
  1. 读取数据规格
  2. 创建采样用 generator
  3. 循环生成各批噪声
  4. 若是 pMF，按样本 ID 循环分配类别标签，保证类别均衡
  5. 用 `generate_samples` 生成图像
  6. 反归一化并逐张写 PNG

### `main`

- 作用：
  FID 脚本总入口。
- 输入/输出：
  - 输入：CLI
  - 输出：无
- 内部流程：
  1. 解析参数并加载 run 配置
  2. 用 CLI 覆盖采样相关配置
  3. 检查数据集和 pMF 样本数约束
  4. 决定设备和 checkpoint
  5. 设置随机种子
  6. 构造模型与优化器
  7. 恢复 checkpoint
  8. 创建 `fid/` 目录
  9. 调 `generate_fid_samples`
  10. 调 `calculate_metrics`
  11. 写 `metrics_*.json`
  12. 需要时删除样本目录
  13. 打印最终 FID 和文件位置
- 阅读注意点：
  - 这里构造优化器不是为了继续训练，而是因为 `restore_checkpoint` 需要把优化器状态也一并恢复

## 测试文件

### `pMF/tests/test_muon.py`

这个仓库目前只看到一组显式单测，集中验证 Muon 更新逻辑。

### `identity_zeropower`

- 作用：
  测试替身函数，让 `zeropower_via_newtonschulz5` 退化成恒等映射。

### `MuonUpdateTests.test_muon_update_uses_standard_momentum_buffer`

- 作用：
  验证 `muon_update` 确实按“标准 momentum buffer”更新，而不是别的变体。

### `MuonUpdateTests.test_optimizer_step_matches_standard_nesterov_momentum`

- 作用：
  验证优化器一步更新后，参数结果与预期 Nesterov 动量公式一致。

## 建议的阅读顺序

如果你准备真正读代码，推荐顺序如下：

1. 先读本文件前两节，搞清楚两条主线和术语。
2. 读 `meanflow.py` 的 `TrainConfig -> build_model -> build_loss -> train`。
3. 回头读 `MeanFlowLoss` 和 `PixelMeanFlowLoss`，理解训练目标。
4. 再读 `Unet`、`MeanFlowTransformer`、`PmfTransformer` 三个模型。
5. 如果你关心独立推理链，再去读 `pmf.py -> pmfDiT.py`。
6. 最后补 `evaluate.py`、`fid_eval.py`、`fidelity_wrapper.py`。

## 一句话总结

这个项目最核心的代码事实是：

- `meanflow.py` 是训练与实验平台
- `pmf.py + pmfDiT.py` 是独立推理封装
- `MeanFlowLoss` 和 `PixelMeanFlowLoss` 决定了两条方法线的训练差异
- `PmfTransformer` / `pmfDiT` 都体现了“图像式输出，速度空间监督”的 pMF 思路
