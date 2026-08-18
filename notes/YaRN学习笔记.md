# YaRN 学习笔记

> 对应代码：`model/model.py` 中的 `precompute_freqs_cis` 函数
> 目标：搞懂"用只训练过 2k 上下文的模型，去服务 32k 超长输入"是怎么做到的。

---

## 0. 起点：原始代码有哪些问题

原始 `precompute_freqs_cis` 有三处致命问题，导致函数根本无法运行：

### 0.1 频率初始化行语法畸形

```python
freqs, attn_factor=(1.0/(rope_base **(torch.arange(0, dim, 2)[:dim//2].float()/dim), 1.0),1.0)
```

解析后右值是 `(1.0 / (某个元组), 1.0)`，即 **浮点数除以元组**，运行时会抛 `TypeError: unsupported operand type(s) for /: 'float' and 'tuple'`。

正确写法应是：

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
attn_factor = 1.0
```

### 0.2 缺少导入

函数签名用了 `Optional[dict]`，但 `Optional` 没导入 → 定义函数时就 `NameError`；YaRN 还要用 `math.log`，`math` 也没导入。需补：

```python
import math
from typing import Optional
```

### 0.3 attention 因子公式错误

原代码：

```python
if end > orig_max:
    attn_factor = attn_factor * (end / orig_max) ** (beta_fast / beta_slow)
```

代入实际值 `end=32768, orig_max=2048, beta_fast=32, beta_slow=1`：

```
attn_factor = 16 ** 32 ≈ 1e38
```

attention 会被瞬间放大到天文数字而崩溃。真正的 YaRN 用的是**对数级**补偿 `0.1·ln(scale)+1`，不是指数级。

---

## 1. 从零理解 RoPE：钟的类比

### 1.1 一面钟记录位置

RoPE（旋转位置编码）的本质：**给句子里每个 token 发一面钟，指针角度 = 它的位置**。

- 第 0 个词 → 0°
- 第 1 个词 → 30°
- 第 2 个词 → 60°
- ……

当两个 token 互相做注意力时，比较指针角度差，就知道彼此隔多远。

### 1.2 一面钟为什么不够（撞车）

如果每词转 30°，那么：

- 第 0 个词 → 0°
- 第 12 个词 → 360° → **又回到 0°**

第 0 个词和第 12 个词角度差为 0，模型会以为它俩在同一位置——乱套。这就是钟的**周期性撞车**问题。

### 1.3 很多面转速不同的钟

解决办法：用**一排钟**，转速从快到慢排开。例如：

| 词的位置 | 快钟(每词30°) | 慢钟(每词1°) |
|---|---|---|
| 第 0 个词 | 0° | 0° |
| 第 12 个词 | 0° ← 撞了 | 12° ← 没撞 |

- **快钟**：管"隔 1 个词还是隔 2 个词"这种**近距离精细分辨**；
- **慢钟**：管"隔很远还是同一位置"这种**远距离不撞车**。

模型里其实有几十面钟（本例 32 面），转速从最快排到最慢。代码里的 `freqs` 就是这一排钟的**转速表**。

### 1.4 为什么不能靠"圈数计数器"

一个自然想法：钟撞车是因为只看角度，那再加一个"转了几圈"的计数器不就行了？

**不行，因为旋转记不住圈数**：旋转 30° 和旋转「30° + 360°」让向量停在**同一个位置**，物理上完全一样，那一圈是"白转"的，没有任何痕迹留下。所以圈数这个信息**存不进旋转里**。

如果把圈数单独存成一个数字喂给模型，那就变成了**绝对位置编码**（直接告诉模型"这是第 32768 个词"），而绝对位置编码恰恰有外推会崩的毛病——模型没见过 32768 这个数，照样懵。

**关键洞察**：用户想要的"能区分远距离的东西"，其实由**最慢的几面钟**充当了——它们转得极慢、还没转完一圈，所以在很长一段距离内每个角度都独一无二，天然就是粗略的远距离分辨员。所以 RoPE 用"一排转速不同的钟"来顶替"圈数计数器"。

---

## 2. 长上下文的痛点：训练长度

### 2.1 模型只见过 0~2048

训练时模型只读过最多 2048 个词的句子。推理时要处理 32768 个词。问题：每种钟在训练时"见过哪些角度"是固定的，超出就会遇到没见过的角度。

核心判据只有一个：

> **训练时，这面钟的指针有没有把"整个表盘"都走遍过？**

模型像个死记硬背的学生：训练见过哪些角度就只会处理哪些，没见过的角度一出现就懵。

### 2.2 快钟无所谓

假设快钟每 4 个词转完一整圈，训练 2048 个词：

```
2048 ÷ 4 = 512 圈
```

训练期间这面钟把表盘上**每个角度都转了 500 多遍**。推理到第 32768 个词，它转出的角度模型都见过 500 次了——✅ 熟，不懵。

### 2.3 慢钟会懵

假设某面慢钟每 6000 个词才转完一整圈：

- 训练（0~2048）：`2048 ÷ 6000 ≈ 0.34 圈`，指针只从 0° 挪到约 **123°**。模型只认得 0°~123°。
- 推理第 32768 个词：`32768 ÷ 6000 ≈ 5.46 圈`，指针跑到约 **166°**——超出 123°，模型没见过，❌ 懵了。

### 2.4 一句话总结

> 同样从 2048 走到 32768：**快钟**因为训练时早转了上百圈，再多转无所谓；**慢钟**因为训练时才挪了不到一圈，一往外走就进了没见过的角度，模型抓瞎。

---

## 3. YaRN 的解法

### 3.1 核心招数：慢钟降速 factor 倍

把慢钟的转速**除以 16**。原来 6000 词转一圈，现在变成 `6000 × 16 = 96000` 词转一圈。再看第 32768 个词：

```
32768 ÷ 96000 ≈ 0.34 圈 ≈ 123°
```

指针又落回 **123°**——正是训练时见过的角度。✅ 不懵了。

### 3.2 为什么偏偏是除以 16

降速 16 倍后，慢钟在第 32768 个词的角度等于：

```
32768 × (原速 ÷ 16) = (32768 ÷ 16) × 原速 = 2048 × 原速
```

而 `2048 × 原速` 正是慢钟在**训练时第 2048 个词**的角度——模型 100% 见过。

> **除以 16，相当于把"第 32768 个词"在慢钟眼里伪装成"第 2048 个词"。**

这个 16 = 32768 ÷ 2048，就是配置里的 `factor`。让所有新位置在慢钟看来都落在 0~2048 这个它认识的范围内。

### 3.3 代价：分辨率下降（不是撞车）

**纠正一个常见误解**：降速后 32768 和 2048 并不会"撞车"。用数字验证（原速 0.06°/词，降速 16 倍后 0.00375°/词）：

- 第 2048 个词：`2048 × 0.00375 = 7.68°`
- 第 32768 个词：`32768 × 0.00375 = 122.88°`

两者角度不同，没撞。真正代价是**分辨率下降 16 倍**：

| | 每往后 1 个词，指针挪动 |
|---|---|
| 降速前 | 0.06° |
| 降速后 | 0.00375°（小了 16 倍） |

原来差 1 个词就能看出角度差，现在得差 16 个词才看得出同样的角度差。慢钟变"糊"了，更不擅长分辨近距离。

**为什么能接受**：分辨"离得很近的位置"本来就不是慢钟的活，是快钟的活。快钟没动，还是满分辨率在转。慢钟的职责是"管远距离、不撞车"，降速后照常履行，只是近距离变糊——而近距离有快钟兜底，所以整体不塌。

### 3.4 边界：有限倍外推

`factor` 不能无限放大。这里用 16 倍（2k→32k）很稳；硬上 100 倍，慢钟会糊到没边、快钟外推也撑不住，质量明显下降。**YaRN 是"优雅地外推一个有限的倍数"，不是"任意延长"。**

---

## 4. ramp 旋钮：三段平滑过渡

### 4.1 为什么不能硬切

有 32 面钟，如果硬划一条线"0~14 号不动，15~31 号降速 16 倍"——相邻两面钟一个满速一个突然慢 16 倍，断崖会让模型不适应。

### 4.2 旋钮形状

YaRN 给每面钟一个 **0 到 1 之间的旋钮值** `ramp`：

- 旋钮 = 0 → 原速转（纯外推，快钟）
- 旋钮 = 1 → 降速 16 倍（纯插值，慢钟）
- 旋钮 = 0.5 → 折中，降速 8 倍（中间钟）

旋钮值随钟从快到慢**从 0 平滑升到 1**，三段：

```
旋钮值
 1 │             ***************  ← 慢钟区(纯插值,降速16倍)
   │            /
   │           /   ← 中间区(平滑过渡)
   │          /
 0 │**********                      ← 快钟区(纯外推,原速)
   └────────────────────────────── 钟的索引
     0      5       14           31
          low      high
```

- 索引 0~(low-1)：旋钮=0，原速（最快的几面，训练时早把表盘走遍）
- 索引 low~high：旋钮从 0 升到 1，逐渐降速
- 索引 (high+1)~31：旋钮=1，降速 16 倍（最慢的几面）

### 4.3 low / high 分界怎么定

本例算出 **`low=5, high=14`**，由代码 `find_correction_range` 计算。

### 4.4 beta_fast / beta_slow 的含义

判断一面钟"够不够快"，用它在训练长度 2048 里**转了几圈**：

- **`beta_fast = 32`**：转了 ≥32 圈的钟，算"快"，原速不动。
- **`beta_slow = 1`**：转了 ≤1 圈的钟，算"慢"，必须降速。
- 转了 1~32 圈之间的，就是中间过渡区。

`find_correction_dim` 干的活，就是把"32 圈""1 圈"这两个门槛**换算成钟的索引号**（5 和 14）。本质是把"转几圈"翻译成"第几面钟"。

### 4.5 混合公式

对第 i 面钟：

```
实际转速 = 原速 × (1 - 旋钮[i])  +  原速/16 × 旋钮[i]
```

对应代码：

```python
freqs = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1 - ramp)
```

- 快钟（旋钮=0）→ `原速 × 1 + 0 = 原速` ✅
- 慢钟（旋钮=1）→ `0 + 原速/16 = 原速/16` ✅
- 中间（旋钮=0.5）→ `一半原速 + 一半降速` ✅

---

## 5. attention 补偿因子

YaRN 发现外推后 attention 的"音量"会漂移，需要乘个补偿系数。

- 配置里显式给了 `attention_factor` 就直接用（本例是 `1.0`，不补偿）。
- 没给就用经验公式 `0.1 × ln(放大倍数) + 1`，是**对数级的小调整**。

这正好替代了原来会爆炸到 1e38 的 `(end/orig_max)**(beta_fast/beta_slow)`。

---

## 6. 完整代码逐行对照

```python
def precompute_freqs_cis(dim, end=32*1024, rope_base=1000000, rope_scaling=None):
```

### 块 1：一排钟的转速表

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
attn_factor = 1.0
```

- `dim`：单个头的完整维度（如 64）。
- `freqs`：算出一排钟的转速，长度 `dim//2 = 32`。左边大（快钟）、右边小（慢钟）。
- `rope_base=1000000`：底数 θ，越大整体转得越慢。
- `end=32768`：要处理的序列长度。
- `attn_factor=1.0`：注意力补偿因子初值。
- `rope_scaling=None` 时不走 YaRN，就是普通 RoPE。

### 块 2：读 YaRN 配置

```python
if rope_scaling is not None:
    orig_max, factor, beta_fast, beta_slow = (
        rope_scaling["original_max_position_embeddings"],  # 训练长度 2048
        rope_scaling["factor"],                            # 降速倍数 16
        rope_scaling["beta_fast"],                         # 快钟门槛 32 圈
        rope_scaling["beta_slow"],                         # 慢钟门槛 1 圈
    )
```

### 块 3：三个辅助函数（圈数 → 钟索引）

```python
def find_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def find_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)

def linear_ramp_mask(low, high, n):
    if low == high:
        high += 0.001  # 防除零，让斜坡至少有点宽度
    linear = (torch.arange(n, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear, 0, 1)
```

- `find_correction_dim`：输入"转 N 圈"，输出"这是第几面钟"。
- `find_correction_range`：把 32 圈、1 圈换成索引区间 `[low, high]`；`floor`/`ceil` 包住过渡区，`max`/`min` 防越界。
- `linear_ramp_mask`：就是那个**旋钮**，在 `[low, high]` 间画 0→1 斜坡。

### 块 4：算出分界 low/high

```python
low, high = find_correction_range(beta_fast, beta_slow, dim, rope_base, orig_max)
# 代入配置算出 low=5, high=14
```

### 块 5：两种转速 + 旋钮混合

```python
inv_freq_extrapolation = freqs          # 原速（快钟用）
inv_freq_interpolation = freqs / factor # 降速 16 倍（慢钟用）

ramp = linear_ramp_mask(low, high, dim // 2)
freqs = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1 - ramp)
```

- `inv_freq_extrapolation`：原速，对应"快钟不动"。
- `inv_freq_interpolation = freqs / factor`：除以 16，对应"慢钟降速，把 32768 伪装回 2048"。
- 最后一行：每面钟按旋钮值在两种转速间加权平均。**走完这行，`freqs` 已是每面钟实际该用的新转速。**

### 块 6：注意力补偿

```python
if "attention_factor" in rope_scaling:
    attn_factor = rope_scaling["attention_factor"]   # 本例 1.0
else:
    scale = end / orig_max if end > orig_max else 1.0
    attn_factor = 0.1 * math.log(scale) + 1.0        # 对数级补偿
```

### 块 7：组装成最终结果

```python
t = torch.arange(end, device=freqs.device, dtype=torch.float32)
freqs = torch.outer(t, freqs) * attn_factor
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
return freqs_cis
```

- `t = arange(end)`：位置序列 `[0, 1, ..., 32767]`，每个词的编号。
- `torch.outer(t, freqs)`：位置 × 转速 = **角度矩阵**，形状 `(32768, 32)`，第 `[pos, i]` 个元素是"第 pos 个词在第 i 面钟上的指针角度"。
- `* attn_factor`：乘补偿因子。
- `torch.polar(1, 角度)`：把角度变成**复数** `cosθ + i·sinθ`（模长 1，在单位圆上），即"指针指向哪个方向"的数学表示。
- 返回 `freqs_cis` 给下游 `apply_rotary_emb`，用它把 query/key 向量旋转到对应角度。

---

## 7. 避坑指南

1. **`dim` 的语义**：本函数里 `dim` 是**单个头的完整维度**（如 64），`freqs` 长度是 `dim//2`。HuggingFace 内部常把"已折半的 dim"到处传，两套约定混看会算错 2 倍边界。

2. **复数 `.angle()` 会绕回 `(-π, π]`**：调试时看到 `16.0` 变 `-2.85` 别慌，那是同一个角 mod 2π。比较旋转结果要用 `cos/sin` 或 `.real/.imag`，别直接比 angle。

3. **`torch.polar` 要 float**：`freqs` 必须是 `float32/64`，传 half 会报错；bf16 训练时通常这里算 fp32 再转。

4. **别在 `forward` 里重复算 `freqs_cis`**：它是"位置→常量"的纯函数，应在模型 `__init__` 时算一次缓存为 `self.register_buffer`，否则每个 batch 都重算 `outer` 很浪费。

5. **`attn_factor` 千万别写成指数式**：`(end/orig_max)**(beta_fast/beta_slow)` 会爆炸到 1e38 毁掉 attention。要用对数级 `0.1·ln(scale)+1`。

---

## 8. 关键数字速查（MokioMindConfig 默认值）

| 参数 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 512 | 隐藏维度 |
| `num_attention_heads` | 8 | 注意力头数 |
| `dim` (head_dim) | 64 | 单头维度 = 512/8 |
| 钟的数量 | 32 | = dim//2 |
| `rope_theta` (`rope_base`) | 1000000 | 底数 θ |
| `original_max_position_embeddings` | 2048 | 训练时最大长度 |
| `factor` | 16 | 慢钟降速倍数 |
| `end` | 32768 | 推理序列长度 = 2048×16 |
| `beta_fast` | 32 | "转几圈算快钟"门槛 |
| `beta_slow` | 1 | "转几圈算慢钟"门槛 |
| `attention_factor` | 1.0 | attention 补偿（1.0=不补偿） |
| `low` / `high` | 5 / 14 | 快/慢分界索引（算出） |

**验证结果**（实跑）：
- 输出 `freqs_cis` 形状 `(32768, 32)`，`torch.complex64` ✅
- 低频维（idx=31）：`yarn角度 × 16 == 普通RoPE角度`（精确到 6 位小数）→ 插值按 factor 缩放正确 ✅
- 高频维（idx=0）：YaRN 与普通 RoPE 的 `cos/sin` 完全一致 → 外推未缩放正确 ✅
- `|freqs_cis| == 1` → 复数在单位圆上 ✅

---

## 9. 进阶思考与延伸

### 9.1 当前方案的局限性

- 频率切分边界是**静态**的——`low/high` 只由 `dim/θ/orig_max` 决定，跟实际推理长度 `end` 无关。若 `end` 远超 `factor·orig_max`（比如配 factor=16 却跑到 100k），低频维插值会过度压缩，长距离精度下降。
- `attention_factor=1.0` 是固定值，不随 `end` 自适应。

### 9.2 扩展：Dynamic YaRN

让 `factor` 随当前 `end` 动态计算 `factor = max(end/orig_max, 1.0)`，切分区间和 attn_factor 也跟着重算。HuggingFace LLaMA 3.1 的 `llama3` 类型即此思路（多了 `low_freq_factor`/`high_freq_factor` 两个平滑边界）。改动点：把 `factor` 从配置读改成动态算，其余逻辑不变。

### 9.3 推荐后续学习

- **NTK-aware Scaling**：YaRN 的前身，只用一个公式整体缩放频率、不分区。理解它能看出 YaRN"为什么要分频"。
- **`partial_rotary_factor`**：不是所有头维度都上 RoPE，只旋转一部分（GPT-NeoX 风格），可降低计算量。
- **`apply_rotary_emb`**：消费 `freqs_cis` 的下游函数，用复数乘法旋转向量 `q_rotated = q_complex * freqs_cis`，是 RoPE 链路的最后一块拼图。

---

## 10. 一句话总览

> 一个只会读 2k 词的模型，靠"一排转速不同的钟 + 对慢钟降速伪装 + 中间平滑过渡"，就能稳稳处理 32k 长文，还基本分得清谁离谁多远——这就是 YaRN。本质是"快钟外推、慢钟插值、中间平滑"，代价是慢钟分辨率下降，但由快钟兜底所以无伤大雅；它是有限倍的优雅外推，不是无限续杯。
