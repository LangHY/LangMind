import torch

X = torch.ones(2, 4, 8)
print(X)

W_q = torch.ones(8, 8)
W_k = torch.ones(8, 8)
W_v = torch.ones(8, 8)

Q = X @ W_q
K = X @ W_k
V = X @ W_v


Q = Q.view(2, 4, 4, 2) # 4 Qhead
K = K.view(2, 4, 2, 4) # 2 KVhead
V = V.view(2, 4, 2, 4)
print(f"Q = {Q}")
print(f"K = {K}")
print(f"V = {V}")
K = K.repeat_interleave(2, dim=1)
V = V.repeat_interleave(2, dim=1)



print(f"Q = {Q}")
print(f"K = {K}")
print(f"V = {V}")