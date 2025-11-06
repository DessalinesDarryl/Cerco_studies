# src/xai/attention.py
# Placeholder: si tu veux récupérer l'attention vraie, il faut hooker nn.TransformerEncoderLayer.
def extract_dummy_attention(model, X):
    # retourne une "matrice" identité moyenne pour MVP
    import torch
    B,T,F = X.shape
    return torch.eye(F).unsqueeze(0).repeat(B,1,1)
