import os
import sys
import yaml
import torch
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import HybridBPModel
from src.dataset import get_dataloaders
from src.train import train
from src.evaluate import evaluate, plot_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config('configs/config.yaml')
    seed = cfg['training']['seed']
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg, seed)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = HybridBPModel(cfg).to(device)

    # Trigger LazyLinear init
    dummy_scal = torch.zeros(1, 1, 128, 128).to(device)
    dummy_ppg = torch.zeros(1, 1, 1250).to(device)
    _ = model(dummy_scal, dummy_ppg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    model, history = train(model, train_loader, val_loader, cfg, device)

    os.makedirs('results', exist_ok=True)
    plot_loss(history, save_path='results/loss_curve.png')

    metrics = evaluate(model, test_loader, device, scaler_dir='checkpoints', out_dir='results')
    print("\n=== Final Test Results ===")
    for bp in ['sbp', 'dbp']:
        m = metrics[bp]
        print(f"{bp.upper()} → MAE: {m['mae']:.2f} | Std: {m['std']:.2f} | R²: {m['r2']:.3f}")


if __name__ == '__main__':
    main()
