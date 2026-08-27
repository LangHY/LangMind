#为什么需要位置编码
# 文本的先后顺序也包含信息，attention 机制只是单纯的相乘
# 相对位置编码

# YaRN

import torch
print("=====torch.where=====")
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([10, 20, 30, 40, 50])

condition = x > 3
print(condition)
result = torch.where(condition, x, y)
# Return a tensor of elements selected from either input or other, depending on condition.
# 条件过滤：condition=True时x对应的元素保留，condition=False时x对应的元素由 y对应的替代
print(result)


print("=====torch.arange=====")
# start, end, step -> tensor
t = torch.arange(0, 10, 2)
print(t)

t = torch.arange(5, 0, -1)
print(t)


print("=====torch.outer=====")
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
result = torch.outer(v1, v2)
# 外积：用v1的每个元素乘v2 
print(result)


print("=====torch.cat=====")
t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])
t2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])
print(t1.shape)
print(t1)
result = torch.cat((t1, t2), dim=0)
print(f"shape = {result.shape}\ncat第 0 维 = {result}")

result = torch.cat((t1, t2), dim=1)
print(f"shape = {result.shape}\ncat第 1 维 = {result}")

result = torch.cat((t1, t2), dim=-1)
print(f"shape = {result.shape}\ncat第 2 维（最后一维） = {result}")
print("=====unsqueeze=====")
t1 = torch.tensor([1, 2, 3])
t2 = t1.unsqueeze(dim=0)
# 在dim前扩展一维
print(t1.shape)
print(t2)
print(t2.shape)

print("==========")

x = torch.arange(1, 9)
# x = torch.randn(2, 3, 3)

theta = torch.tensor(0.5)
def RoPE(x:torch.Tensor, theta:torch.Tensor):
    
    even = x[0::2]
    odd = x[1::2]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    print(x)
    # print(x[0]) # 打印第一个维度的第零个元素

    rotated_even = even * cos_theta - odd * sin_theta
    rotated_odd = even * sin_theta + odd * cos_theta
    # 分组（一奇加一偶）

    result = torch.empty_like(x)
    # 设定一个跟x形状相同的空张量
    result[0::2] = rotated_even
    result[1::2] = rotated_odd
    # 奇偶重新插入
    return result

print(RoPE(x, theta=torch.tensor(0.5)))