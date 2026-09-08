import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_token, d_ff):  
        super().__init__()
        # 定义升维与降维的线性层
        self.up = nn.Linear(d_token, d_ff)
        self.down = nn.Linear(d_ff, d_token)
        
    
    def forward(self, x):
        # 门控：筛选特征
        gate = F.silu(self.up(x))
        
        content = self.up(x)
        # 升维后逐元素相乘
        x = gate * content
        # 降维
        return self.down(x) 