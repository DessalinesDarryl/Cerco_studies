# src/models/xaimodel.py
import torch, torch.nn as nn
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, depth, num_classes):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, batch_first=True)
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.attn_last = None
    def forward(self, x):
        # x: (B, T, F) — ici on suppose T=1 → (B,1,F)
        z = self.proj_in(x); z = self.enc(z); self.attn_last=None
        logits = self.cls(z.mean(dim=1))
        return logits
