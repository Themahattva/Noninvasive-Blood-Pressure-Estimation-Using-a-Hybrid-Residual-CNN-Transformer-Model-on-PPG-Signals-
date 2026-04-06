"""
Visualizes multi-head attention weights over a PPG window.
Helps identify which temporal regions the model focuses on.

Usage: python src/attention_viz.py --checkpoint checkpoints/best_model.pt --sample_id 0
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import HybridBPModel
from src.utils import load_config


class AttentionExtractor(nn.Module):
    """Wraps PPG branch to capture attention weights."""
    def __init__(self, model):
        super().__init__()
        self.branch = model.ppg_branch
        self._attn_weights = None

    def forward(self, ppg):
        b = self.branch
        x = torch.relu(b.conv(ppg))
        x = x.permute(0, 2, 1)
        x = b.proj(x)
        x = b.pos_enc(x)
        x = b.norm1(x)
        # need_weights=True to get attention map
        attn_out, attn_weights = b.attn(x, x, x, need_weights=True, average_attn_weights=False)
        self._attn_weights = attn_weights.detach().cpu()  # (B, heads, T, T)
        return attn_out


def load_sample(cfg, sample_id):
    ppg = np.load(os.path.join(cfg['data']['ppg_dir'], f"{sample_id:06d}.npy"))
    scal = cv2.imread(os.path.join(cfg['data']['scalogram_dir'], f"{sample_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
    ppg_t = torch.from_numpy(ppg.reshape(1, 1, -1)).float()
    scal_t = torch.from_numpy(scal.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return ppg_t, scal_t, ppg.flatten()


def plot_attention(ppg_raw, attn_weights, save_path=None):
    """
    attn_weights: (heads, T, T) — plot mean across query dimension
    """
    n_heads = attn_weights.shape[0]
    T = attn_weights.shape[-1]
    t = np.linspace(0, 10, T)

    fig, axes = plt.subplots(n_heads + 1, 1, figsize=(14, 3 * (n_heads + 1)))

    # Plot raw PPG
    axes[0].plot(t, ppg_raw[:T], color='steelblue', linewidth=0.8)
    axes[0].set_title('Raw PPG Signal (normalized)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim([0, 10])

    # Plot per-head attention (mean over query positions)
    for h in range(n_heads):
        attn_h = attn_weights[h].mean(axis=0).numpy()  # (T,)
        attn_norm = (attn_h - attn_h.min()) / (attn_h.max() - attn_h.min() + 1e-8)
        ax = axes[h + 1]
        ax.fill_between(t, attn_norm, alpha=0.5, color=f'C{h}')
        ax.plot(t, attn_norm, color=f'C{h}', linewidth=0.7)
        ax.set_title(f'Attention Head {h + 1}')
        ax.set_ylabel('Weight')
        ax.set_xlim([0, 10])

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Multi-Head Self-Attention over PPG Window', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention plot: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--sample_id', type=int, default=0)
    parser.add_argument('--out', default='results/attention_viz.png')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cpu')

    model = HybridBPModel(cfg)
    dummy_scal = torch.zeros(1, 1, 128, 128)
    dummy_ppg = torch.zeros(1, 1, 1250)
    _ = model(dummy_scal, dummy_ppg)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    ppg_t, scal_t, ppg_raw = load_sample(cfg, args.sample_id)
    extractor = AttentionExtractor(model)

    with torch.no_grad():
        extractor(ppg_t)

    attn = extractor._attn_weights[0]  # (heads, T, T)
    plot_attention(ppg_raw, attn, save_path=args.out)


if __name__ == '__main__':
    main()
