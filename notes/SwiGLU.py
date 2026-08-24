import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_token, d_ff):  
        super().__init__()

        self.up = nn.Linear(d_token, d_ff)
        self.down = nn.Linear(d_ff, d_token)
        
    
    def forward(self, x):
        gate = F.silu(self.up(x))
        content = self.up(x)

        x = gate * content
        return self.down(x) 