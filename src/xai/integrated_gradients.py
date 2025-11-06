# src/xai/integrated_gradients.py
import torch
from captum.attr import IntegratedGradients
def compute_ig(model, inputs, target_idx, steps=50):
    model.eval(); ig = IntegratedGradients(model)
    return ig.attribute(inputs, target=target_idx, n_steps=steps)
