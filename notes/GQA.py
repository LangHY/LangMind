import torch
import torch.nn as nn
import math


print("=====nn.Dropout=====")
dropout_layer = nn.Dropout(p=0.5)
# 以 0.5的概率清零某个元素，防止过拟合
t1 = torch.arange(2, 5, dtype=torch.float32)
print(f"t1 = {t1}")
for i in range(4):
    t2 = dropout_layer(t1)
    print(t2)
#为了保持期望不变，未清零的元素会扩大


print("=====nn.Linear=====")
# 线性变化，就是对张量乘权重矩阵再加上偏置
# 权重和偏置由优化器优化
layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.randn(1, 3)
t2 = torch.randn(3)
# t1 多了一层 batch 维的"外壳"，把每个数字包成了长度为 1 的向量。
print(f"t1 = {t1}")
print(f"t2 = {t2}")
print(layer(t1))
print(layer(t2))


print("=====view=====")
t = torch.randn(2, 6)
print(f"t = {t}")
t_view1 = t.view(3, 4) #形状改为(3, 4)
print(t_view1)
t_view2 = t.view(4, 3)
print(t_view2)
# 可以改变形状的前提是总元素个数不变, (2,2)不能变成(3, 3),但可以变成(1, 4)


print("=====transpose=====")
t1 = torch.randn(2, 3)
print(f"t1 = {t1}")
print(t1.transpose(0, 1))
# 交换第 0 维和第 1 维


print("=====torch.triu=====")
x = torch.randn(3, 3)
print(f"x = {x}")
print(torch.triu(x))
# 转换为下三角矩阵
print(torch.triu(x, diagonal=-1))
# 将对角线下移一格
print(torch.triu(x, diagonal=1))
# 将对角线上移一格

print("=====GQA实现=====")

X = torch.randn(2, 4, 8) #输入 4 个 token，每个 token 8 维
w_Q = torch.randn(8, 8)
w_K = torch.randn(8, 8)
w_V = torch.randn(8, 8)


def Attention(X:torch.tensor, w_Q:torch.tensor, w_K:torch.tensor, w_V:torch.tensor, nums_head:int):
    # 计算输入 token 的 QKV
    Q = torch.tensor(X @ w_Q)
    K = torch.tensor(X @ w_K)
    V = torch.tensor(X @ w_V)
    print(f"Q_shape = {Q.shape}")
    print(f"K_shape = {K.shape}")
    print(f"V_shape = {V.shape}")
    print(f"K^T = {K.transpose(-1, -2).shape}")


    seq = (Q.shape[-2]) # token数量
    batch = Q.shape[0] #batch数
    d_token = X.shape[-1] #token的维度（用于计算注意力头分得的维度）
    d_K = list(K.shape)[-1] #K的维度（ 用于计算 attention）
    #依据注意力头数计算每个注意力头分得的维度（Head QKV 的维度）
    d_Head = int(d_token / nums_head)


    # 依据注意力头数拆 token
    Q = Q.view(batch, seq, nums_head, d_Head)
    K = K.view(batch, seq, nums_head, d_Head)
    # [batch, seq, nums_head, head_dim]
    # [2,     4,   2,      4]
    print(Q.shape, K.shape)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    print(Q.shape, K.shape)
    # [batch, nums_head, seq, head_dim]
    # PyTorch 的批量矩阵乘法：
    # 默认会把：
    # 最后两个维度当矩阵乘法维度
    # 前面的维度当 batch 维度。
    # 所以我们希望：

    # 前面的维度：
    # batch
    # head

    # 最后两个：
    # token
    # dim

    score = Q @ (K.transpose(-1, -2))
    print(f"score = {score.shape}")
    # [batch, nums_head, query token, key token]
    score = score / math.sqrt(d_K)

    attention = (torch.softmax(score, dim=-1)) @ V
    return attention


result = Attention(X, w_Q, w_K, w_V, nums_head=2)
print(result)