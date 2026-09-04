from math import sqrt

import torch

def attn(X:torch.Tensor):
    # 获取输入的形状参数
    batch = X.shape[0]
    seq = X.shape[-2]
    dim = X.shape[-1]

    # 引入多头注意力
    num_head = 4
    # 计算每个头分配的 dim
    d_head = dim // num_head

    # 引入 GQA：只减少 KV 头的"数量"，每个头的维度 d_head 与 Q 头保持一致
    num_kv = 2

    # 定义权重：K/V 投影只输出 num_kv 个头，矩阵更小——这正是 GQA 省参数省显存的来源
    # GQA = 保持每个头的“带宽”（d_head）不变，只减少 K/V 表征的“份数”（num_kv）
    W_q = torch.randn(dim, num_head * d_head)
    W_k = torch.randn(dim, num_kv * d_head)
    W_v = torch.randn(dim, num_kv * d_head)

    # 计算 QKV
    Q = X @ W_q # 2x4x16 (16 = 4头 x 4维)
    K = X @ W_k # 2x4x8  (8 = 2头 x 4维)
    V = X @ W_v # 2x4x8

    # 拆分token
    # 关键的是view必须按照前两个参数为batch类，后两个参数为形状类，否则形状错误，语义错乱
    # transpose 负责把num_head和seq对调，以head为组织单位，让多个注意力头放在一起计算
    Q = Q.view(batch, seq, num_head, d_head).transpose(1, 2) # 4 q_head
    K = K.view(batch, seq, num_kv, d_head).transpose(1, 2) # 2 kv_head，每头依然是 d_head 维
    V = V.view(batch, seq, num_kv, d_head).transpose(1, 2) # 2 kv_head
    # transpose前：2 4 2 4 表示 2batch 每个batch 4token 每个token分为2 head 每个 head 4 维


    # 打印形状
    print("QKV形状")
    print(Q.shape)
    print(K.shape)
    print(V.shape)
    
    # 此时的Q KV 形状对不上 需要对 KV 张量进行扩充
    group_size = num_head // num_kv
    K = K.repeat_interleave(group_size, dim=1) #沿dim将每个元素连续复制group_size次（每个元素紧挨着自己复制）
    V = V.repeat_interleave(group_size, dim=1) #指定在第1维，也就是 num_kv对应的维度进行复制
    print("QKV形状")
    print(Q.shape)
    print(K.shape)
    print(V.shape)


    # 计算注意力分数
    attention = (torch.softmax((Q @ K.transpose(-1, -2)) / sqrt(d_head), dim=-1)) @ V 
    # 这里的 K 不能用 K.T 会将 2 4 8变为 8 4 2
    print(f"before concating: {attention.shape}")

    # 并行计算后合并
    attention = attention.transpose(1, 2) # 将所有 token 的所有head结果放在一起，因此需要再变换回去
    attention = attention.reshape(batch, seq, num_head * d_head)

    # transpose 的本质是改变“我们当前以谁为组织单位看待数据”

 
    return attention

X = torch.randn(2, 4, 16) #2 batch 4 tokens 16 dimensions

print(attn(X).shape)

"""
Attention 计算时，把 Head 放在 Token 前面，
使张量以 Head 为组织单位，每个 Head 都包含完整的 token 序列；
Attention 完成后，
再把 Token 放到 Head 前面，使张量以 Token 为组织单位，
将同一个 token 在不同 Head 中得到的表示拼接起来。 
"""

"""
torch.repeat_interleave(
    x,          # 要复制的张量
    repeats,    # 每个元素复制几次
    dim=1       # 沿哪个维度复制
)
"""