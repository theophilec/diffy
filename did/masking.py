"""
masking.py

Implements mask generation for images, relying on the Blackman window.
"""
import torch


def generate_mu_pt(patch):
    h, w = patch.shape[-2:]
    H_h = torch.blackman_window(h).unsqueeze(1)
    H_w = torch.blackman_window(w).unsqueeze(1)
    H = H_h.mm(H_w.T)
    return H / H.max()