"""
Ablation study: trains 3 model variants and compares performance.
  1. ScalogramOnly  — CNN branch only
  2. PPGOnly        — Transformer branch only  
  3. Hybrid (full)  — proposed model

Usage: python src/ablation.py --config configs/config.yaml
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ScalogramBranch, PPGBranch
from src.dataset import get_dataloaders
from src.train import train
from src.evaluate import get_predictions, inverse_transform, compute_metrics
import joblib


class ScalogramOnlyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        self.cnn = ScalogramBranch(m['dropout_cnn'], m['dense_units'])
        self.out = nn.Linear(m['dense_units'], 2)

    def forward(self, scalogram, ppg):
        return self.out(self.cnn(scalogram))


class PPGOnlyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        self.ppg = PPGBranch(m['num_heads'], m['key_dim'], m['dense_units'])
        self.out = nn.Linear(m['dense_units'], 2)

    def forward(self, scalogram, ppg):
        return self.out(self.ppg(ppg))


class BiLSTMBaseline(nn.Module):
    """Standard BiLSTM baseline for comparison with proposed Transformer branch."""
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        hidden = m['dense_units']
        self.conv = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bilstm = nn.LSTM(64, hidden // 2, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(m['dropout_fusion'])
        )
        self.out = nn.Linear(hidden, 2)

    def forward(self, scalogram, ppg):
        # ppg: (B, 1, 1250)
        x = torch.relu(self.conv(ppg))   # (B, 64, 1250)
        x = x.permute(0, 2, 1)           # (B, 1250, 64)
        x, _ = self.bilstm(x)            # (B, 1250, hidden)
        x = x[:, -1, :]                  # last step
        return self.out(self.fc(x))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_variant(name, model, train_loader, val_loader, test_loader, cfg, device):
    print(f"\n{'='*50}")
    print(f"  Variant: {name}")
    print(f"{'='*50}")

    # Init lazy layers
    dummy_scal = torch.zeros(1, 1, 128, 128).to(device)
    dummy_ppg = torch.zeros(1, 1, 1250).to(device)
    _ = model(dummy_scal, dummy_ppg)

    ckpt = f"checkpoints/ablation_{name.lower().replace(' ', '_')}.pt"
    cfg_copy = {**cfg, 'training': {**cfg['training'], 'checkpoint_path': ckpt}}

    model, _ = train(model, train_loader, val_loader, cfg_copy, device)
    preds, labels = get_predictions(model, test_loader, device)

    sbp_scaler = joblib.load('checkpoints/sbp_scaler.pkl')
    dbp_scaler = joblib.load('checkpoints/dbp_scaler.pkl')
    sbp_pred, dbp_pred, sbp_true, dbp_true = inverse_transform(preds, labels, sbp_scaler, dbp_scaler)

    sbp_mae, sbp_std, sbp_r2 = compute_metrics(sbp_true, sbp_pred, 'SBP')
    dbp_mae, dbp_std, dbp_r2 = compute_metrics(dbp_true, dbp_pred, 'DBP')
    return {
        'model': name,
        'sbp_mae': sbp_mae, 'sbp_std': sbp_std, 'sbp_r2': sbp_r2,
        'dbp_mae': dbp_mae, 'dbp_std': dbp_std, 'dbp_r2': dbp_r2,
    }


def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    from src.model import HybridBPModel
    variants = [
        ('Scalogram Only', ScalogramOnlyModel(cfg).to(device)),
        ('PPG Only (Transformer)', PPGOnlyModel(cfg).to(device)),
        ('BiLSTM Baseline', BiLSTMBaseline(cfg).to(device)),
        ('Hybrid (Proposed)', HybridBPModel(cfg).to(device)),
    ]

    results = []
    for name, model in variants:
        r = run_variant(name, model, train_loader, val_loader, test_loader, cfg, device)
        results.append(r)

    df = pd.DataFrame(results)
    print("\n\n=== ABLATION RESULTS ===")
    print(df.to_string(index=False))
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/ablation_results.csv', index=False)


if __name__ == '__main__':
    main()
