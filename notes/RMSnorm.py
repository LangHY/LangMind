# Norm:归一化，使数据的平均值为 0，标准差为一
# 计算梯度时：损失函数对权重求偏导
# 输出公式：Y = W * X
# 权重更新公式：W_new = W_old - lr * dL/dW_old
# 梯度：dL/dW = dL/dY * dY/dW = dL/dY * X （链式法则）
# 也就是说，梯度与本身的值 X 有关
# X 过大或者过小都容易导致梯度爆炸或者消失（链式法则决定了要更新浅层权重必须连乘所有导数）
# RMSNorm: 归一化，相比 Norm 少了均值计算
import torch
# 对张量开方求倒数
t = torch.rsqrt(torch.tensor([[4.0, 9.0], [8.0, 5.0]]))
print(t)

# 创建全 1 张量
t = torch.ones(3, 4)
print(t)

