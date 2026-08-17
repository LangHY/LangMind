# from turtle import forward

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
# RMSNorm需要继承nn.module类（神经网络层）
class RMSNorm(nn.Module):
#__init__
    def __init__(self, dim:int, eps:float=1e-5):
    # 传入维度以及伊普西隆
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 创建可学习参数：权重
        self.weight = nn.Parameter(torch.ones(dim))

# _norm
    def _norm(self, x):
        eps = self.eps
        pow_mean = x.pow(x).mean(-1, keepdim=True)
        result = torch.rsqrt(pow_mean + eps)
        #返回缩放因子
        return result

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x) * x
        # x.float() 先将输入转换成float32精度再计算
        # .type_as(x) 再将x转换为原来的类型
        # 缩放因子乘以输入乘以权重


# YaRN (Yet another RoPE extensioN)：推理时把训练阶段学到的短上下文 RoPE 外推到更长序列
def precompute_freqs_cis(dim:int, end:int=32*1024, rope_base:int=1000000, rope_scaling:Optional[dict] = None):
    """
    YaRN (Yet another RoPE extensioN)：用只训练过 2k 上下文的模型，去服务 32k 超长输入。

    =========================== 钟的类比（理解 YaRN 的钥匙）===========================
    RoPE = 给每个 token 发"一排转速不同的钟"，指针角度 = 它的位置。
      - 快钟：转得快，管"近距离精细分辨"，但很快转满一圈会撞车；
      - 慢钟：转得慢，管"远距离不撞车"，但训练时可能连一圈都没转完。
    多面钟组合，既能精细分辨近距离、又能不撞车地记住远距离。
    注：旋转角度记不住"转了几圈"（转 360° 等于没转），所以不能靠圈数计数器，
        而是用"一排转速不同的钟"来顶替——最慢的几面天然充当远距离分辨员。

    =========================== 为什么要 YaRN ===========================
    训练只见过 0~orig_max(2048) 的位置：
      - 快钟：到 2048 早转了上百圈，整圈表盘走遍了 → 再往外转无所谓(外推 extrapolation)；
      - 慢钟：到 2048 才挪了表盘一小块 → 一往外走就指到没见过的角度，模型懵了。
    YaRN 的解法：对慢钟"降速 factor(16) 倍"，把第 32768 个词在慢钟眼里伪装成第 2048 个
      (因为 32768 × 原速/16 = 2048 × 原速，正是训练末端见过的角度)，让模型只见到熟悉角度；
      快钟不动；中间的钟平滑过渡。
    代价：慢钟分辨率掉 16 倍(变糊)，但近距离精细分辨靠快钟兜底，所以无伤大雅。
    边界：factor 不能无限放大，过大慢钟会糊到没边、快钟外推也撑不住——YaRN 是"有限倍优雅外推"。

    =========================== 步骤 ↔ 代码 对照 ===========================
      1. 一排钟的转速表            freqs = 1.0 / (rope_base ** (...))        # 见下"1)"
      2. 读 YaRN 配置              orig_max, factor, beta_fast, beta_slow    # 见下 if rope_scaling
      3. 圈数→钟索引的翻译器        find_correction_dim / find_correction_range
      4. 算出快/慢分界 low,high    low, high = find_correction_range(...)
      5. 慢钟降速 factor 倍        inv_freq_interpolation = freqs / factor
      6. ramp 旋钮平滑过渡          freqs = interpolation*ramp + extrapolation*(1-ramp)
      7. attention 对数级补偿       attn_factor = ...   # 注意是对数级，不是指数！
      8. 位置 × 转速 = 角度         freqs = torch.outer(t, freqs) * attn_factor
      9. 角度 → 复数指针 e^{iθ}     freqs_cis = torch.polar(ones, freqs)

    =========================== 避坑 ===========================
      - dim 是"单个头的完整维度"(如 64)，freqs 长度是 dim//2；别和 HF 内部"已折半的 dim"混淆。
      - 复数 .angle() 会绕回 (-π, π]：调试时 16.0 显示成 -2.85 是同一个角 mod 2π，比较旋转用 cos/sin。
      - torch.polar 要求 float；bf16 训练时这里算 fp32 再转。
      - freqs_cis 是"位置→常量"，应在 __init__ 算一次缓存(register_buffer)，别每个 batch 重算。
      - attn_factor 千万别写成 (end/orig_max)**(beta_fast/beta_slow)，会爆炸到 1e38 毁掉 attention。

    配置示例(MokioMindConfig): factor=16, orig_max=2048, beta_fast=32, beta_slow=1,
                              end=32768, dim=64 → 共 32 面钟, low=5, high=14。
    """
    # 1) 基础逆频率：freqs[i] = 1 / rope_base^(2i/dim)
    #    i 小 → freqs 大 → 波长短(高频维)；i 大 → freqs 小 → 波长长(低频维)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],  # 训练时的最大长度
            rope_scaling["factor"],                            # 期望放大的倍数
            rope_scaling["beta_fast"],                         # 高频边界(旋转圈数)
            rope_scaling["beta_slow"],                         # 低频边界(旋转圈数)
        )

        # ---- YaRN 核心：按"旋转圈数"把频率维度切成三段，分别用不同策略 ----
        # find_correction_dim：某维在训练长度内正好转完 num_rotations 圈，反解出该维索引
        def find_correction_dim(num_rotations, dim, base, max_pos):
            return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        # find_correction_range：把 [beta_fast, beta_slow] 两个圈数阈值映射成维度区间 [low, high]
        def find_correction_range(low_rot, high_rot, dim, base, max_pos):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_pos))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_pos))
            return max(low, 0), min(high, dim - 1)

        # linear_ramp_mask：在 [low, high] 之间生成 0→1 的线性斜坡
        #   索引 < low → 0(高频维，走外推)；索引 > high → 1(低频维，走插值)
        def linear_ramp_mask(low, high, n):
            if low == high:
                high += 0.001  # 避免除零，保证斜坡至少有一点点宽度
            linear = (torch.arange(n, dtype=torch.float32) - low) / (high - low)
            return torch.clamp(linear, 0, 1)

        low, high = find_correction_range(beta_fast, beta_slow, dim, rope_base, orig_max)

        # 外推(extrapolation)：高频维直接沿用训练时的旋转速度
        # 插值(interpolation)：低频维把旋转速度按 factor 压缩，等价于位置按 factor 放大
        inv_freq_extrapolation = freqs
        inv_freq_interpolation = freqs / factor

        ramp = linear_ramp_mask(low, high, dim // 2)
        # ramp≈0 的高频维 → 用 extrapolation；ramp≈1 的低频维 → 用 interpolation
        freqs = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1 - ramp)

        # 注意力缩放因子：YaRN 发现外推后 attention 幅值会漂移，需要补偿
        # 配置里显式给了 attention_factor 就直接用，否则用经验公式 0.1*ln(scale)+1
        if "attention_factor" in rope_scaling:
            attn_factor = rope_scaling["attention_factor"]
        else:
            scale = end / orig_max if end > orig_max else 1.0
            attn_factor = 0.1 * math.log(scale) + 1.0

    # 2) 位置序列 t = [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # 3) 角度矩阵 angles[pos, i] = pos * freqs[i]，再乘上注意力因子
    freqs = torch.outer(t, freqs) * attn_factor
    # 4) 转成复数 e^{i*angle}(实部=cos, 虚部=sin)，供 apply_rotary_emb 用复数乘法旋转
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis