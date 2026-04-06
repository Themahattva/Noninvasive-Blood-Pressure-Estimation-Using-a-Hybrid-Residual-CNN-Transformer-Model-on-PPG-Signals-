import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import joblib


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for scal, ppg, label in loader:
        scal, ppg = scal.to(device), ppg.to(device)
        pred = model(scal, ppg).cpu().numpy()
        all_preds.append(pred)
        all_labels.append(label.numpy())
    return np.vstack(all_preds), np.vstack(all_labels)


def inverse_transform(preds, labels, sbp_scaler, dbp_scaler):
    sbp_pred = sbp_scaler.inverse_transform(preds[:, 0:1]).flatten()
    dbp_pred = dbp_scaler.inverse_transform(preds[:, 1:2]).flatten()
    sbp_true = sbp_scaler.inverse_transform(labels[:, 0:1]).flatten()
    dbp_true = dbp_scaler.inverse_transform(labels[:, 1:2]).flatten()
    return sbp_pred, dbp_pred, sbp_true, dbp_true


def compute_metrics(true, pred, label='SBP'):
    mae = np.mean(np.abs(true - pred))
    std = np.std(true - pred)
    r2 = r2_score(true, pred)
    print(f"{label} | MAE: {mae:.2f} mmHg | Std: {std:.2f} | R²: {r2:.3f}")
    return mae, std, r2


def bland_altman_plot(true, pred, label, save_path=None):
    mean = (true + pred) / 2
    diff = true - pred
    md = np.mean(diff)
    sd = np.std(diff)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean, diff, alpha=0.3, s=5)
    ax.axhline(md, color='gray', linestyle='-', label=f'Mean diff: {md:.2f}')
    ax.axhline(md + 1.96 * sd, color='red', linestyle='--', label=f'+1.96SD: {md + 1.96*sd:.2f}')
    ax.axhline(md - 1.96 * sd, color='red', linestyle='--', label=f'-1.96SD: {md - 1.96*sd:.2f}')
    ax.set_xlabel('Mean (True + Pred) / 2 (mmHg)')
    ax.set_ylabel('Difference (True - Pred) (mmHg)')
    ax.set_title(f'Bland-Altman Plot: {label}')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def scatter_plot(true, pred, label, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, alpha=0.3, s=5)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'r--', label='Identity')
    ax.set_xlabel(f'True {label} (mmHg)')
    ax.set_ylabel(f'Predicted {label} (mmHg)')
    ax.set_title(f'True vs Predicted {label}')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_loss(history, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Huber)')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(model, test_loader, device, scaler_dir='checkpoints', out_dir='results'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    sbp_scaler = joblib.load(os.path.join(scaler_dir, 'sbp_scaler.pkl'))
    dbp_scaler = joblib.load(os.path.join(scaler_dir, 'dbp_scaler.pkl'))

    preds, labels = get_predictions(model, test_loader, device)
    sbp_pred, dbp_pred, sbp_true, dbp_true = inverse_transform(preds, labels, sbp_scaler, dbp_scaler)

    sbp_mae, sbp_std, sbp_r2 = compute_metrics(sbp_true, sbp_pred, 'SBP')
    dbp_mae, dbp_std, dbp_r2 = compute_metrics(dbp_true, dbp_pred, 'DBP')

    scatter_plot(sbp_true, sbp_pred, 'SBP', os.path.join(out_dir, 'scatter_sbp.png'))
    scatter_plot(dbp_true, dbp_pred, 'DBP', os.path.join(out_dir, 'scatter_dbp.png'))
    bland_altman_plot(sbp_true, sbp_pred, 'SBP', os.path.join(out_dir, 'ba_sbp.png'))
    bland_altman_plot(dbp_true, dbp_pred, 'DBP', os.path.join(out_dir, 'ba_dbp.png'))

    return {
        'sbp': {'mae': sbp_mae, 'std': sbp_std, 'r2': sbp_r2},
        'dbp': {'mae': dbp_mae, 'std': dbp_std, 'r2': dbp_r2}
    }
