# from sympy.multipledispatch.conflict import consistent
from transformers import PretrainedConfig


# 规定模型架构配置
# huggingface transformer 中的类
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

# RMSNorm
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
# RMSNorm需要继承nn.module类（神经网络层）
# class RMSNorm(nn.Module):
# #__init__
#     def __init__(self, dim:int, eps:float=1e-5):
#     # 传入维度以及伊普西隆
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         # 创建可学习参数：权重
#         self.weight = nn.Parameter(torch.ones(dim))

# # _norm
#     def _norm(self, x):
#         eps = self.eps
#         pow_mean = x.pow(x).mean(-1, keepdim=True)
#         result = torch.rsqrt(pow_mean + eps)
#         #返回缩放因子
#         return result

#     def forward(self, x):
#         return self.weight * self._norm(x.float()).type_as(x) * x
#         # x.float() 先将输入转换成float32精度再计算
#         # .type_as(x) 再将x转换为原来的类型
#         # 缩放因子乘以输入乘以权重


# # YaRN (Yet another RoPE extensioN)：推理时把训练阶段学到的短上下文 RoPE 外推到更长序列
# def precompute_freqs_cis(dim:int, end:int=32*1024, rope_base:int=1000000, rope_scaling:Optional[dict] = None):
#     """
#     YaRN (Yet another RoPE extensioN)：用只训练过 2k 上下文的模型，去服务 32k 超长输入。

#     =========================== 钟的类比（理解 YaRN 的钥匙）===========================
#     RoPE = 给每个 token 发"一排转速不同的钟"，指针角度 = 它的位置。
#       - 快钟：转得快，管"近距离精细分辨"，但很快转满一圈会撞车；
#       - 慢钟：转得慢，管"远距离不撞车"，但训练时可能连一圈都没转完。
#     多面钟组合，既能精细分辨近距离、又能不撞车地记住远距离。
#     注：旋转角度记不住"转了几圈"（转 360° 等于没转），所以不能靠圈数计数器，
#         而是用"一排转速不同的钟"来顶替——最慢的几面天然充当远距离分辨员。

#     =========================== 为什么要 YaRN ===========================
#     训练只见过 0~orig_max(2048) 的位置：
#       - 快钟：到 2048 早转了上百圈，整圈表盘走遍了 → 再往外转无所谓(外推 extrapolation)；
#       - 慢钟：到 2048 才挪了表盘一小块 → 一往外走就指到没见过的角度，模型懵了。
#     YaRN 的解法：对慢钟"降速 factor(16) 倍"，把第 32768 个词在慢钟眼里伪装成第 2048 个
#       (因为 32768 × 原速/16 = 2048 × 原速，正是训练末端见过的角度)，让模型只见到熟悉角度；
#       快钟不动；中间的钟平滑过渡。
#     代价：慢钟分辨率掉 16 倍(变糊)，但近距离精细分辨靠快钟兜底，所以无伤大雅。
#     边界：factor 不能无限放大，过大慢钟会糊到没边、快钟外推也撑不住——YaRN 是"有限倍优雅外推"。

#     =========================== 步骤 ↔ 代码 对照 ===========================
#       1. 一排钟的转速表            freqs = 1.0 / (rope_base ** (...))        # 见下"1)"
#       2. 读 YaRN 配置              orig_max, factor, beta_fast, beta_slow    # 见下 if rope_scaling
#       3. 圈数→钟索引的翻译器        find_correction_dim / find_correction_range
#       4. 算出快/慢分界 low,high    low, high = find_correction_range(...)
#       5. 慢钟降速 factor 倍        inv_freq_interpolation = freqs / factor
#       6. ramp 旋钮平滑过渡          freqs = interpolation*ramp + extrapolation*(1-ramp)
#       7. attention 对数级补偿       attn_factor = ...   # 注意是对数级，不是指数！
#       8. 位置 × 转速 = 角度         freqs = torch.outer(t, freqs) * attn_factor
#       9. 角度 → 复数指针 e^{iθ}     freqs_cis = torch.polar(ones, freqs)

#     =========================== 避坑 ===========================
#       - dim 是"单个头的完整维度"(如 64)，freqs 长度是 dim//2；别和 HF 内部"已折半的 dim"混淆。
#       - 复数 .angle() 会绕回 (-π, π]：调试时 16.0 显示成 -2.85 是同一个角 mod 2π，比较旋转用 cos/sin。
#       - torch.polar 要求 float；bf16 训练时这里算 fp32 再转。
#       - freqs_cis 是"位置→常量"，应在 __init__ 算一次缓存(register_buffer)，别每个 batch 重算。
#       - attn_factor 千万别写成 (end/orig_max)**(beta_fast/beta_slow)，会爆炸到 1e38 毁掉 attention。

#     配置示例(MokioMindConfig): factor=16, orig_max=2048, beta_fast=32, beta_slow=1,
#                               end=32768, dim=64 → 共 32 面钟, low=5, high=14。
#     """
#     # 1) 基础逆频率：freqs[i] = 1 / rope_base^(2i/dim)
#     #    i 小 → freqs 大 → 波长短(高频维)；i 大 → freqs 小 → 波长长(低频维)
#     freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     attn_factor = 1.0

#     if rope_scaling is not None:
#         orig_max, factor, beta_fast, beta_slow = (
#             rope_scaling["original_max_position_embeddings"],  # 训练时的最大长度
#             rope_scaling["factor"],                            # 期望放大的倍数
#             rope_scaling["beta_fast"],                         # 高频边界(旋转圈数)
#             rope_scaling["beta_slow"],                         # 低频边界(旋转圈数)
#         )

#         # ---- YaRN 核心：按"旋转圈数"把频率维度切成三段，分别用不同策略 ----
#         # find_correction_dim：某维在训练长度内正好转完 num_rotations 圈，反解出该维索引
#         def find_correction_dim(num_rotations, dim, base, max_pos):
#             return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

#         # find_correction_range：把 [beta_fast, beta_slow] 两个圈数阈值映射成维度区间 [low, high]
#         def find_correction_range(low_rot, high_rot, dim, base, max_pos):
#             low = math.floor(find_correction_dim(low_rot, dim, base, max_pos))
#             high = math.ceil(find_correction_dim(high_rot, dim, base, max_pos))
#             return max(low, 0), min(high, dim - 1)

#         # linear_ramp_mask：在 [low, high] 之间生成 0→1 的线性斜坡
#         #   索引 < low → 0(高频维，走外推)；索引 > high → 1(低频维，走插值)
#         def linear_ramp_mask(low, high, n):
#             if low == high:
#                 high += 0.001  # 避免除零，保证斜坡至少有一点点宽度
#             linear = (torch.arange(n, dtype=torch.float32) - low) / (high - low)
#             return torch.clamp(linear, 0, 1)

#         low, high = find_correction_range(beta_fast, beta_slow, dim, rope_base, orig_max)

#         # 外推(extrapolation)：高频维直接沿用训练时的旋转速度
#         # 插值(interpolation)：低频维把旋转速度按 factor 压缩，等价于位置按 factor 放大
#         inv_freq_extrapolation = freqs
#         inv_freq_interpolation = freqs / factor

#         ramp = linear_ramp_mask(low, high, dim // 2)
#         # ramp≈0 的高频维 → 用 extrapolation；ramp≈1 的低频维 → 用 interpolation
#         freqs = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1 - ramp)

#         # 注意力缩放因子：YaRN 发现外推后 attention 幅值会漂移，需要补偿
#         # 配置里显式给了 attention_factor 就直接用，否则用经验公式 0.1*ln(scale)+1
#         if "attention_factor" in rope_scaling:
#             attn_factor = rope_scaling["attention_factor"]
#         else:
#             scale = end / orig_max if end > orig_max else 1.0
#             attn_factor = 0.1 * math.log(scale) + 1.0

#     # 2) 位置序列 t = [0, 1, ..., end-1]
#     t = torch.arange(end, device=freqs.device, dtype=torch.float32)
#     # 3) 角度矩阵 angles[pos, i] = pos * freqs[i]，再乘上注意力因子
#     freqs = torch.outer(t, freqs) * attn_factor
#     # 4) 转成复数 e^{i*angle}(实部=cos, 虚部=sin)，供 apply_rotary_emb 用复数乘法旋转
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs_cis

# def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
#     bs, slen, num_key_value_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
    
#     return (
#             x[:,:,:,None, :]
#             .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
#             .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
#             )
# def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple:
#     """将 precompute_freqs_cis 预计算的旋转角作用到 Q/K 上（复数乘法 = 旋转）"""
#     # [bs, seq, head, dim] -> [bs, seq, head, dim/2, 2(实部,虚部)] -> 复数张量
#     # 相邻两维配成一对 (x1+i*x2)，旋转时它们一起转，信息不丢
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     # freqs_cis: [seq, dim/2] -> [1, seq, 1, dim/2]，广播到所有 batch 和所有头
#     freqs_cis = freqs_cis[None, : xq.shape[1], None, :]
#     # 复数相乘 = 角度相加 = 旋转；再转回实数并拍平最后一维
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


# class Attention(nn.Module):
#     """分组查询注意力 (Grouped-Query Attention, GQA)

#     ┌──────────────────────────────────────────────────────────────────┐
#     │ 方案    │ Q 头数 │ KV 头数 │ KV cache 大小 │ 质量                 │
#     ├──────────────────────────────────────────────────────────────────┤
#     │ MHA     │   8    │   8     │  1x (基准)    │ 基准                 │
#     │ MQA     │   8    │   1     │  1/8          │ 明显掉点             │
#     │ GQA     │   8    │   2     │  1/4          │ 接近 MHA（本实现）   │
#     └──────────────────────────────────────────────────────────────────┘
#     类比：8 位编辑(Q头)都要查资料库(K/V 头)
#       - MHA：每人配一个专属资料员 → 又快又准，但养 8 个人太贵
#       - MQA：全公司只请 1 个资料员 → 省钱，但忙不过来、质量下降
#       - GQA：每 4 位编辑共享 1 名资料员（共 2 名）→ 省钱且质量几乎无损
#     LLaMA-2 70B / LLaMA-3 全系都采用 GQA。
#     """
#     def __init__(self, args: MokioMindConfig):
#         super().__init__()

#         # 兼容未配置 kv 头数的情况：退化为标准 MHA
#         self.num_key_value_heads = (
#             args.num_key_value_heads
#             if args.num_key_value_heads is not None
#             else args.num_attention_heads
#         )
#         # Q 头数必须能被 KV 头数整除，否则分组不均、repeat 后头数对不上
#         assert (
#             args.num_attention_heads % self.num_key_value_heads == 0
#         ), "num_attention_heads must be divisible by num_key_value_heads"

#         self.n_local_heads = args.num_attention_heads              # Q 头数: 8
#         self.n_local_kv_heads = self.num_key_value_heads          # KV 头数: 2
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 组大小: 4 个 Q 头共享 1 个 KV 头
#         self.head_dim = args.hidden_size // args.num_attention_heads  # 单头维度: 512 / 8 = 64

#         # Q 投影输出全部 8 个头；K/V 投影只输出 2 个头 → 参数量与激活值都降为 1/4
#         self.wq = nn.Linear(args.hidden_size, self.n_local_heads * self.head_dim, bias=False)
#         self.wk = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
#         self.wv = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
#         self.wo = nn.Linear(self.n_local_heads * self.head_dim, args.hidden_size, bias=False)

#         self.attn_dropout = args.dropout
#         self.flash_attention = args.flash_attention

#     def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         bs, seq_len, _ = x.shape  # [batch, seq_len, hidden_size]

#         # 1) 线性投影并"切头"：Q 切成 8 头，K/V 只切成 2 头 —— GQA 从这里就开始省
#         xq = self.wq(x).view(bs, seq_len, self.n_local_heads, self.head_dim)     # [bs, s, 8, 64]
#         xk = self.wk(x).view(bs, seq_len, self.n_local_kv_heads, self.head_dim)  # [bs, s, 2, 64]
#         xv = self.wv(x).view(bs, seq_len, self.n_local_kv_heads, self.head_dim)  # [bs, s, 2, 64]

#         # 2) 只对 Q/K 施加 RoPE 旋转（V 不携带位置信息，不需要旋转）
#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

#         # 3) GQA 核心：把 2 个 KV 头各复制 4 份"撑"成 8 个，与 Q 头数对齐
#         #    分组规则是"相邻为一组"：Q头 0~3 用 KV头0，Q头 4~7 用 KV头1
#         xk = repeat_kv(xk, self.n_rep)  # [bs, s, 2, 64] -> [bs, s, 8, 64]
#         xv = repeat_kv(xv, self.n_rep)

#         # 4) [bs, seq, head, dim] -> [bs, head, seq, dim]，让 matmul 在后两维 (seq, dim) 上做
#         xq = xq.transpose(1, 2)
#         xk = xk.transpose(1, 2)
#         xv = xv.transpose(1, 2)

#         if self.flash_attention:
#             # PyTorch 融合算子：因果掩码、缩放、softmax 全在算子内部高效完成（FlashAttention 后端）
#             output = torch.nn.functional.scaled_dot_product_attention(
#                 xq, xk, xv,
#                 is_causal=True,
#                 dropout_p=self.attn_dropout if self.training else 0.0,
#             )
#         else:
#             # 手写注意力路径：QK^T / sqrt(d_k) + 因果掩码 + softmax + 加权 V
#             scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
#             if mask is not None:
#                 scores = scores + mask  # mask 形如 [1, 1, seq, seq]，靠广播对齐所有头
#             # softmax 前转 float32 防止 bf16 下溢出（与 RMSNorm 里的 .float() 同理）
#             scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#             scores = F.dropout(scores, p=self.attn_dropout if self.training else 0.0)
#             output = torch.matmul(scores, xv)

#         # 5) [bs, head, seq, dim] -> [bs, seq, head*dim=512]，再投影回 hidden_size
#         output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
#         return self.wo(output)


# class rmsnorm(nn.Module):
#     def __init__(self, hidden_size:int, eps=1e-5):
#         super().__init__()

#         self.eps = eps
#         self.weight = nn.Parameter( # 表示self.weight是模型参数 参与反向传播 
#             torch.ones(hidden_size) #初始化全部设为 1 不改变归一化的结果
#         )
#         # weight 允许模型重新学习每个 hidden dimension 应该放大还是缩小。

#     def forward(self, x:torch.Tensor):
#         rmsnorm = math.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
#         # dim=-1 也就是沿着列变化的方向求平均，也就是沿着行求平均，实际上也就是沿着每个 token 的维度求平均，求得是以 token 为组织单位的平均
#         # keepdim 决定求平均之后是否保留消掉的维度, true -> 本应消失的维度设为 1
#         x = (x / rmsnorm) * self.weight
        
#         return x


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size)
        )
        
    def forward(self, x):
        return (x / torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)) * self.weight


class RoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 计算每个head的维度
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 引入seq
        self.max_position_embeddings = (
            config.max_position_embeddings
        )
        # 引入theta
        self.rope_theta = config.rope_theta

        self.rope_scaling = config.rope_scaling

        # 计算频率
        # rope_theta相当于是一个基准角度，决定不同维度对的频率如何分布
        inv_freq = 1 / (self.rope_theta ** (
           2 * range(0, self.head_dim, 2, dtype=torch.float32) /  #float32保证精度
           self.head_dim
        ))
        # 存储中间计算结果
        self.register_buffer( # 不需要参与训练，但也是模型权重的一部分，与Parameters相对
            "inv_freq",
            inv_freq,
            persistent=False # 不持久存储
        )
        # 计算位置参数 创建所有 position 只用于角度计算公式
        position = torch.range(self.max_position_embeddings, dtype=torch.float32)
        # 角度=位置 x 频率
        theta = torch.outer(position, inv_freq)
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        # 保存中间结果
        # shape:[max_position_embeddings, D/2] 也就是每个位置的每个维度对的结果
        self.register_buffer(
            "cos_cached",
            cos,
            persistent=False # 只在运行时使用，不保存
        )
        self.register_buffer(
            "sin_cached",
            sin,
            persistent=False 
        )
    
    def forward(self, q:torch.Tensor, k:torch.Tensor, position_ids:torch.Tensor):
        # 引入position_ids：[B, S] 表示每个token 的绝对位置
        # 取出每个 token 对应的角度
        cos = self.cos_cached[position_ids].squeeze(dim=1)
        sin = self.sin_cached[position_ids].squeeze(dim=1)
        # shape:[B, S, D/2]某个batch某个seq的某个维度对的正余弦
        # 为了广播要挤出一个head维度
        # shape:[B, 1, S, D/2]

        # 拆分
        even_q = q[..., 0::2]
        odd_q = q[..., 1::2]

        even_k = k[..., 0::2]
        odd_k = k[..., 1::2]

        # 旋转
        q = torch.stack(
            [even_q * cos - odd_q * sin, even_q * sin + odd_q * cos],
            dim = -1
        ).flatten(-2, -1)

        k = torch.stack(
            [even_k * cos - odd_k * sin, even_k * sin + odd_k * cos],
            dim = -1
        ).flatten(-2, -1)

        return q, k


class gqa(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.group_size = self.num_attention_heads // self.num_kv_heads
        # self.training = config.training
        self.dropout = config.dropout
        # 通过线性层来获取 QKV 权重，自动变为可训练参数
        self.w_q = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=False
        )
        self.w_k = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_kv_heads,
            bias=False
        )
        self.w_v = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_kv_heads,
            bias=False
        )
        self.w_o = nn.Linear(
            self.hidden_size, #输入的维度是多个head拼起来的结果
            self.hidden_size, #经过线性层使不同head的信息充分混合（按照每个head学习到的特征加权）
            bias=False
        )

    def forward(self, x:torch.Tensor):
        batch, seq, dim= x.shape

        
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        query = query.view(batch, seq, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.group_size, dim=1)
        value = value.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.group_size, dim=1)

        causal_mask = torch.triu( #创建上三角矩阵
            torch.ones( #形状与
                seq,
                seq,
                dtype=bool,
                device=x.device
            ),
            diagonal=1, #对角线相对于主对角线移动的单位数，主对角线为 0
        )

        score = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        score = score.masked_fill( #对应causal_mask中的 0 和 1,1 对应位置改为最小（-inf）,0 对应位置不变
            causal_mask,
            torch.finfo(score.dtype).min #当前张量数据类型所能表示的最小值
        )

        score = torch.softmax(score, dim=-1) #因为有些元素趋近于-inf，e^-inf = 0,所以就达到了掩码的效果，防止模型看到答案

        #dropout:训练时随机把一部分神经网络产生的信息暂时关掉（置零），防止模型过于依赖某个特征，也就是随机屏蔽一定比例的注意力关系，增加训练随机性
        score = ((F.dropout(
            score,
            p=self.dropout, #置零比例
            training=self.training #只在训练时启用
        )) @ value).transpose(1, 2).reshape(batch, seq, self.num_attention_heads * self.head_dim)
        attention = self.w_o(score) #让不同的head信息充分混合
        return attention

        







        