# src/xai/deeplift.py
import torch
from captum.attr import DeepLift
def compute_deeplift(model, inputs, target_idx):
    model.eval(); dl = DeepLift(model)
    attributions = dl.attribute(inputs, target=target_idx)  # shape = inputs
    return attributions
