import torch
import torch.nn as nn


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