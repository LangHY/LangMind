import torch


X = torch.randn(2, 4, 6, 8)
def rope(X:torch.Tensor):
    batch = X.shape[0]
    head = X.shape[1]
    seq = X.shape[2]
    head_dim = X.shape[3]

    nums_2dims = head_dim // 2
    freqs = 1 / (10000 ** (2*torch.arange(0, head_dim, 2) / head_dim))

    positions = torch.arange(0, seq)

    theta = torch.outer(positions, freqs)

    cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(0)
    sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(0)

    print(cos_theta.shape)
    print(sin_theta.shape)

    odd = X[..., 1::2] #...表示前面的维度全部保留，否则默认在 0 维度做切片操作
    even = X[..., 0::2]

    print(odd.shape)
    print(even.shape)

    rotated_odd = odd * cos_theta + even * sin_theta
    rotated_even = even * cos_theta - odd * sin_theta

    # 将两个张量按照dim进行合并
    result = torch.stack(
        [rotated_even, rotated_odd],
        dim = -1
    ).flatten(-2, -1) #将张量的从-2 到-1 维度进行合并（相乘为一个维度）保持了 QK 的形状
    
    return result

print(rope(X).shape)