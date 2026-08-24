import torch

def ffn(x:torch.Tensor, D_ff:int):
    batch = x.shape[0]
    seq = x.shape[-2]
    dim = x.shape[-1]
    
    # Linear 1
    W1 = torch.randn(dim, D_ff)
    linear1 = x @ W1

    # Relu
    hid = torch.relu(linear1)

    # Linear 2
    W2 = torch.randn(D_ff, dim)
    linear2 = hid @ W2

    print(x.shape)
    print(linear1.shape)
    print(linear2.shape)

x = torch.rand(64, 32, 16)
D_ff = 512
ffn(x, D_ff)