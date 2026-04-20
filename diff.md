# `--attn-impl naive` vs `--attn-impl residual` 完整差异解析

本文档详细对比 `meanflow.py` 中 `--attn-impl naive` 与 `--attn-impl residual` 两种模式在**模型结构、前向传播、训练流程、推理流程、优化器、参数量、显存**等维度上的所有差异。

---

## 一、总体差异概览

| 维度 | `naive` | `residual` |
|---|---|---|
| 残差连接方式 | 显式加法 `x = x + f(x)` | 隐式 attention over depth history |
| 每个 Block 额外模块 | 无 | 2 个 `FullAttentionResidual` |
| 模型顶层额外模块 | 无 | 2 个 `FullAttentionResidual`（u/v output） |
| Block 前向签名 | `forward(x=tensor)` | `forward(history=list[tensor])` |
| 中间状态存储 | 仅保留当前隐状态 | 保留所有历史层输出的列表 |
| 参数量 | 基准 | 多出 `(2 * depth + 2) * hidden_size` 个参数 |
| 显存峰值 | 基准 | 显著增大（history 列表线性增长） |
| 论文出处 | 标准 Pre-LN Transformer | arXiv:2603.15031 Full Attention Residuals |

---

## 二、模型构建差异（`__init__`）

### 2.1 TransformerBlock 构建（meanflow.py:281-294）

**naive 模式：**
```python
self.norm1 = LayerNorm(hidden_size)
self.attn = TransformerAttention(hidden_size, num_heads)
self.norm2 = LayerNorm(hidden_size)
self.mlp = TransformerMlp(hidden_size, mlp_ratio)
self.pre_attn_residual = None    # ← 不创建
self.pre_mlp_residual = None     # ← 不创建
```

**residual 模式：**
```python
self.norm1 = LayerNorm(hidden_size)
self.attn = TransformerAttention(hidden_size, num_heads)
self.norm2 = LayerNorm(hidden_size)
self.mlp = TransformerMlp(hidden_size, mlp_ratio)
self.pre_attn_residual = FullAttentionResidual(hidden_size)   # ← 额外创建
self.pre_mlp_residual = FullAttentionResidual(hidden_size)    # ← 额外创建
```

每个 `FullAttentionResidual` 包含一个 `nn.Parameter(torch.zeros(hidden_size))` 作为 learned pseudo-query。

**每个 TransformerBlock 额外参数：2 × hidden_size**

### 2.2 PmfTransformer 顶层构建（meanflow.py:374-379）

**naive 模式：**
```python
self.u_output_residual = None
self.v_output_residual = None
```

**residual 模式：**
```python
self.u_output_residual = FullAttentionResidual(hidden_size)   # ← 额外创建
self.v_output_residual = FullAttentionResidual(hidden_size)   # ← 额外创建
```

**顶层额外参数：2 × hidden_size**

### 2.3 参数量总差异

以默认配置 `depth=8, hidden_size=256` 为例：

- `shared_depth = depth - head_depth = 8 - 4 = 4`
- `head_depth = max(1, depth // 2) = 4`
- 总 block 数 = 4 (shared) + 4 (u_heads) + 4 (v_heads) = 12
- Block 内额外参数 = 12 × 2 × 256 = **6,144**
- 顶层额外参数 = 2 × 256 = **512**
- **总额外参数 = 6,656**（约 0.006M，相对于整个模型的参数量非常小）

---

## 三、前向传播差异（核心差异）

### 3.1 TransformerBlock.forward（meanflow.py:296-319）

#### naive 模式：标准 Pre-LN + 加法残差

```python
def forward(self, x, history=None):
    attn_out = self.attn(self.norm1(x))      # 1. LayerNorm → Attention
    x = x + attn_out                          # 2. 加法残差
    x = x + self.mlp(self.norm2(x))           # 3. LayerNorm → MLP → 加法残差
    return x, history                          # history 未使用，原样返回 None
```

数据流：
```
x ─────────────────┐
│                   │
├→ LN → Attn ──────+ → x' ──────────────┐
                                         │
x' ────────────────────────────────────┐ │
│                                      │ │
├→ LN → MLP ──────────────────────────+ → x''
```

特点：
- 每层只保留当前隐状态 `x`
- 信息只能通过**逐层累加**传递
- 浅层信息到深层需要经过所有中间层的变换

#### residual 模式：Full Attention Residuals

```python
def forward(self, x=None, history=None):
    # Step 1: 从所有历史层聚合得到 attn 输入
    attn_in = self.pre_attn_residual(history)       # attention over depth
    attn_out = self.attn(self.norm1(attn_in))       # LayerNorm → Attention
    history = [*history, attn_out]                   # attn 输出加入 history

    # Step 2: 从更新后的 history 聚合得到 mlp 输入
    mlp_in = self.pre_mlp_residual(history)          # attention over depth (含 attn_out)
    mlp_out = self.mlp(self.norm2(mlp_in))           # LayerNorm → MLP
    return mlp_out, [*history, mlp_out]              # mlp 输出也加入 history
```

数据流：
```
history = [h₀, h₁, ..., hₙ]
         │
         ├→ FullAttnRes(query_attn) → attn_in    ← 沿 depth softmax 加权求和
         │                              │
         │                     LN → Attn → attn_out
         │                                    │
         ├─── append(attn_out) ──→ history' = [h₀, ..., hₙ, attn_out]
         │                                    │
         │                   FullAttnRes(query_mlp) → mlp_in
         │                                              │
         │                                     LN → MLP → mlp_out
         │                                                   │
         └─── append(mlp_out) ──→ history'' = [h₀, ..., hₙ, attn_out, mlp_out]
```

特点：
- **无显式加法残差**——残差连接完全由 depth-wise attention 隐式实现
- Self-attention 和 MLP 被视为**两个独立的层**，各自产生一个 history 条目
- 每个 sub-layer 可以**直接访问任意历史层**的输出（包括最初的 embedding）
- History 列表线性增长：1 个 block 增加 2 个条目

### 3.2 FullAttentionResidual.forward 详解（meanflow.py:223-229）

```python
def forward(self, history: list[torch.Tensor]) -> torch.Tensor:
    # history: N 个 [B, T, D] 的 tensor
    keys = torch.stack(history, dim=2)                    # [B, T, N, D]
    scores = einsum("d, btnd -> btn", self.query,         # query: [D]（可学习）
                    rms_norm_last_dim(keys, eps=self.eps)) # RMS norm on keys
    weights = softmax(scores, dim=-1)                      # 沿 depth 轴(N) softmax
    return einsum("btn, btnd -> btd", weights, keys)       # 加权求和 → [B, T, D]
```

关键细节：
1. **Query 零初始化** → 初始 softmax 权重均匀分布 = 平均所有历史层 ≈ 加法残差的均匀版本
2. **Softmax 在 float32 精度**计算（`.softmax(dim=-1, dtype=torch.float32)`），然后 cast 回原精度
3. **RMS norm 作用于 keys**（即 history 条目），而非 query
4. **原始值（非 normalized）用于加权求和**——norm 只影响 score 计算

### 3.3 PmfTransformer.forward（meanflow.py:424-469）

#### naive 模式：

```python
tokens = self.build_sequence(x, h, omega, t_min, t_max, labels)

# Shared backbone
for block in self.shared_blocks:
    tokens, _ = block(tokens)                    # 传入 x=tokens

# u 分支
u_tokens = tokens                                # 直接引用（浅拷贝）
for block in self.u_heads:
    u_tokens, _ = block(u_tokens)

# v 分支
v_tokens = tokens                                # 从 shared 输出分叉
for block in self.v_heads:
    v_tokens, _ = block(v_tokens)

# 后处理（两种模式相同）
u_tokens = self.u_norm(u_tokens[:, self.prefix_tokens:])
v_tokens = self.v_norm(v_tokens[:, self.prefix_tokens:])
...
```

#### residual 模式：

```python
tokens = self.build_sequence(x, h, omega, t_min, t_max, labels)

# Shared backbone — 用 history 驱动
shared_history = [tokens]                        # embedding 作为 history[0]
for block in self.shared_blocks:
    _, shared_history = block(history=shared_history)

# u 分支
u_history = list(shared_history)                 # list 浅拷贝（tensor 共享引用）
for block in self.u_heads:
    _, u_history = block(history=u_history)
u_tokens = self.u_output_residual(u_history)     # ← 额外的输出聚合

# v 分支
v_history = list(shared_history)
for block in self.v_heads:
    _, v_history = block(history=v_history)
v_tokens = self.v_output_residual(v_history)     # ← 额外的输出聚合

# 后处理（两种模式相同）
u_tokens = self.u_norm(u_tokens[:, self.prefix_tokens:])
v_tokens = self.v_norm(v_tokens[:, self.prefix_tokens:])
...
```

**关键差异：**

1. **History 初始化**：embedding `tokens` 作为 `history[0]`，所有后续层都可以直接访问原始 embedding
2. **分支拷贝**：`list(shared_history)` 做 list 浅拷贝——两个分支共享 shared 阶段的 tensor 对象，但后续 append 独立
3. **输出聚合**：residual 模式在 head 输出后**额外做一次** `FullAttentionResidual` 聚合，而 naive 直接取最后一层输出

---

## 四、History 列表增长分析

以默认配置 `depth=8`（`shared_depth=4, head_depth=4`）为例：

```
初始:    history = [tokens]                              → len = 1

Shared Block 0:  append(attn_out_0, mlp_out_0)           → len = 3
Shared Block 1:  append(attn_out_1, mlp_out_1)           → len = 5
Shared Block 2:  append(attn_out_2, mlp_out_2)           → len = 7
Shared Block 3:  append(attn_out_3, mlp_out_3)           → len = 9

分叉后 u_history 从 len=9 开始:
u Head Block 0:  append(attn_out, mlp_out)                → len = 11
u Head Block 1:  append(attn_out, mlp_out)                → len = 13
u Head Block 2:  append(attn_out, mlp_out)                → len = 15
u Head Block 3:  append(attn_out, mlp_out)                → len = 17

u_output_residual 聚合: attend over 17 个 [B, T, D] tensor
```

**v 分支同理**，也产生 len=17 的 history。

公式：每个分支的 history 长度 = `1 + 2 * shared_depth + 2 * head_depth = 1 + 2 * depth`

---

## 五、显存差异

### naive 模式
- 前向只保留当前隐状态 `x`：1 × [B, T, D]
- 反向传播需要保存每层的激活用于梯度计算

### residual 模式
- 前向需要保留**完整 history 列表**中所有 tensor：`(1 + 2*depth)` × [B, T, D]
- `torch.stack(history, dim=2)` 在每个 FullAttentionResidual 调用时创建 [B, T, N, D] 张量
- 反向传播需要保存所有 history tensor + stack 操作的中间结果

以默认配置估算（depth=8, B=128, T=64+5=69, D=256, float32）：
- 每个 [B, T, D] tensor：128 × 69 × 256 × 4 bytes ≈ 9 MB
- naive：每层约保存 1 个 ≈ 9 MB
- residual 一个分支的 history（17 个 tensor）：17 × 9 MB ≈ 153 MB
- 加上 stack 和 softmax 的中间张量，residual 的显存开销**显著更大**

---

## 六、优化器交互差异

### Muon 优化器的参数分组（meanflow.py:776-789）

```python
if name.startswith(("shared_blocks.", "u_heads.", "v_heads.")) and param.ndim >= 2 and not name.endswith("bias"):
    muon_params.append(param)       # Muon 更新
else:
    adamw_params.append(param)      # AdamW 更新
```

**naive 模式下**，block 内参数只有：
- `norm1.weight`（1D）→ AdamW
- `attn.qkv.weight`（2D）→ Muon
- `attn.proj.weight`（2D）→ Muon
- `norm2.weight`（1D）→ AdamW
- `mlp.net.0.weight`（2D）→ Muon
- `mlp.net.2.weight`（2D）→ Muon

**residual 模式下**，额外增加：
- `pre_attn_residual.query`（1D，hidden_size）→ **AdamW**（因为 `ndim < 2`）
- `pre_mlp_residual.query`（1D，hidden_size）→ **AdamW**

顶层的 `u_output_residual.query` 和 `v_output_residual.query` 不以 `shared_blocks./u_heads./v_heads.` 开头，也归入 **AdamW**。

**结论：所有 FullAttentionResidual 的 query 参数都由 AdamW 更新**，即使使用 Muon 优化器也是如此。

---

## 七、训练损失计算的差异

训练损失函数 `PixelMeanFlowLoss.__call__`（meanflow.py:621-656）本身**不区分** `attn_impl`——它只调用 `net(...)` 并获取 `(u, v)` 输出。差异完全封装在模型的 `forward` 中。

但 **JVP（Jacobian-Vector Product）计算**（第647行）会受到间接影响：

```python
(u, v), (du_dt, _) = torch.func.jvp(
    warped_u_fn,
    (z_t, t, r),
    (v_c, torch.ones_like(t), torch.zeros_like(r)),
)
```

- `torch.func.jvp` 要求前向计算必须是**纯函数式**的
- residual 模式在前向中创建了大量 list append 和 `torch.stack` 操作
- 这些操作对 JVP 而言需要被正确追踪——list 操作本身不涉及梯度，但 stack 和 einsum 涉及
- 实际影响：residual 模式的 JVP **计算量更大**，因为需要追踪更多的中间操作

---

## 八、推理流程的差异

推理由 `generate_pmf`（meanflow.py:664-683）执行：

```python
for step in range(sample_steps):
    u, _ = model(z_t, t, t - r, omega, t_min, t_max, labels)
    z_t = z_t - (t - r) * u
```

推理中：
- 模型的 `forward` 会按 `attn_impl` 走不同分支（如第三节所述）
- `v` 输出被丢弃（`u, _`），但 residual 模式仍会完整构建 v_history 并聚合
- **推理速度**：residual 模式因为需要维护 history 列表和多次 stack/softmax，每步推理更慢

注意：推理使用 `@torch.no_grad()` 装饰，不需要反向传播，但前向的显存开销差异仍然存在。

---

## 九、梯度流差异（理论层面）

### naive 模式
```
∂L/∂h₀ = ∂L/∂hₙ × ∂hₙ/∂hₙ₋₁ × ... × ∂h₁/∂h₀
```
由于加法残差：`∂hᵢ/∂hᵢ₋₁ = I + ∂f(hᵢ₋₁)/∂hᵢ₋₁`，梯度有**恒等捷径**，缓解梯度消失。

### residual 模式
```
∂L/∂h₀ = Σᵢ (∂L/∂output_i) × (∂output_i/∂h₀)
```
每一层都可以**直接 attend** 到 `h₀`（原始 embedding），梯度可以从任意深度直接回传到第 0 层，理论上提供了比加法残差**更强的梯度捷径**。

但 softmax 归一化意味着浅层权重会被深层条目稀释——这与加法残差的"无衰减直通"不同。

---

## 十、初始化行为差异

### naive 模式
初始状态下每个 block 的 attention 和 MLP 权重随机初始化，加法残差保证初始时信号至少能"穿过"网络（identity shortcut）。

### residual 模式
`FullAttentionResidual.query` 零初始化（`torch.zeros(hidden_size)`），导致：

```python
scores = einsum("d, btnd -> btn", zeros, rms_norm(keys))   # scores 全为 0
weights = softmax(zeros, dim=-1)                             # 均匀分布 1/N
output = mean(history, dim=depth)                            # 退化为均匀平均
```

**初始行为 = 对所有历史层取均匀平均**。随着训练进行，query 学习到不同的注意力模式，逐渐偏离均匀分布。

---

## 十一、完整代码定位索引

| 组件 | 文件位置 | naive 行为 | residual 行为 |
|---|---|---|---|
| 配置字段 | meanflow.py:91 | `attn_impl = "naive"` | `attn_impl = "residual"` |
| CLI 参数 | meanflow.py:955 | `--attn-impl naive` | `--attn-impl residual` |
| `FullAttentionResidual` | meanflow.py:210-229 | 不使用 | 核心模块 |
| `rms_norm_last_dim` | meanflow.py:206-207 | 不使用 | 被 FullAttentionResidual 调用 |
| Block `__init__` 分支 | meanflow.py:289-294 | `pre_attn/mlp_residual = None` | 创建 2 个 FullAttentionResidual |
| Block `forward` 分支 | meanflow.py:301-319 | 加法残差路径 (313-319) | history 路径 (301-311) |
| Model `__init__` 分支 | meanflow.py:374-379 | `u/v_output_residual = None` | 创建 2 个 FullAttentionResidual |
| Model `forward` 分支 | meanflow.py:435-459 | 直接传递 tensor (449-459) | history 列表驱动 (435-448) |
| 优化器分组 | meanflow.py:783 | 无额外 1D 参数 | query 参数归入 AdamW 组 |
| 推理 | meanflow.py:664-683 | 无差异（封装在 forward 中） | 无差异（封装在 forward 中） |
| 损失函数 | meanflow.py:621-656 | 不感知 attn_impl | 不感知 attn_impl |

---

## 十二、总结

`--attn-impl residual` 相比 `naive` 的本质变化是**用 depth-wise attention 替代加法作为残差连接机制**。这带来：

1. **更灵活的跨层信息流**：每层可以自适应地选择从哪些历史层获取信息
2. **更强的梯度直通能力**：任意层可直接回传梯度到 embedding 层
3. **更高的计算和显存开销**：history 线性增长 + 每层额外的 stack/softmax/einsum
4. **极少的额外参数**：仅增加 `(2*depth+2) * hidden_size` 个可学习 query 参数
5. **训练和推理代码路径完全封装在模型内部**：损失函数、采样算法、优化器逻辑（除参数分组）均不需要修改
