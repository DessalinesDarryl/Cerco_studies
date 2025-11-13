# src/models/fusion.py
import torch, torch.nn as nn
class LateFusion(nn.Module):
    def __init__(self, dims, out_dim):
        super().__init__(); self.proj = nn.ModuleList([nn.Linear(d, out_dim) for d in dims])
    def forward(self, xs):
        zs = [p(x) for p,x in zip(self.proj, xs)]
        return sum(zs)/len(zs)
