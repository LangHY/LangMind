from math import sqrt
import torch


Q = torch.rand(3, 3)
K = torch.rand(5, 3)
V = torch.rand(5, 6)
# 获取 K 的维度列表
dk = list(K.shape)
print(f"dk = {dk}")

matrix = Q @ (K.transpose(0, 1)) #转置确保满足矩阵乘法条件
soften = torch.softmax(input=(matrix / sqrt(dk[-1])), dim=-1)
print(f"注意力分数矩阵{soften}")
print(f"注意力分数矩阵形状: {soften.shape}")
attention = soften @ V # @为矩阵乘法
print(attention)