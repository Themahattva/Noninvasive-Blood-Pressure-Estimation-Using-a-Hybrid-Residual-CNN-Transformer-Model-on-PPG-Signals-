import os
import yaml
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_config_snapshot(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'run_config.yaml'), 'w') as f:
        yaml.dump(cfg, f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss(history, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def log_metrics(metrics, epoch=None, prefix=''):
    tag = f"[Epoch {epoch}] " if epoch else ""
    for bp in ['sbp', 'dbp']:
        m = metrics[bp]
        print(f"{tag}{prefix}{bp.upper()} → MAE: {m['mae']:.2f} | Std: {m['std']:.2f} | R²: {m['r2']:.3f}")


def check_bhs_grade(mae, std, label=''):
    """BHS Grade criteria: A: MAE<5 & std<8, B: MAE<8 & std<10, C: MAE<10 & std<12"""
    if mae < 5 and std < 8:
        grade = 'A'
    elif mae < 8 and std < 10:
        grade = 'B'
    elif mae < 10 and std < 12:
        grade = 'C'
    else:
        grade = 'D'
    print(f"BHS Grade {label}: {grade}  (MAE={mae:.2f}, Std={std:.2f})")
    return grade


def check_aami(mae, std, label=''):
    """AAMI standard: MAE <= 5 mmHg, std <= 8 mmHg"""
    passed = mae <= 5.0 and std <= 8.0
    status = 'PASS' if passed else 'FAIL'
    print(f"AAMI {label}: {status}  (MAE={mae:.2f}, Std={std:.2f})")
    return passed
