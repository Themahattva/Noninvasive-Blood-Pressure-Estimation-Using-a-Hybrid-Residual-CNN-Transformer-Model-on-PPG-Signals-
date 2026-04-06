"""
Usage: python main_inference.py --ppg_csv path/to/ppg.csv --checkpoint checkpoints/best_model.pt
CSV must have columns with PPG and ABP signals at 125Hz.
"""
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import HybridBPModel
from src.preprocessing import preprocess_ppg
from src.scalogram import ppg_to_scalogram


def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def predict_from_window(model, ppg_window, cfg, device, sbp_scaler, dbp_scaler):
    scal_cfg = cfg['scalogram']
    scalogram = ppg_to_scalogram(
        ppg_window,
        scales=scal_cfg['scales'],
        wavelet=scal_cfg['wavelet'],
        size=scal_cfg['image_size'],
        gamma=scal_cfg['gamma']
    )
    if scalogram is None:
        return None, None

    scal_tensor = torch.from_numpy(scalogram).unsqueeze(0).unsqueeze(0).float().to(device)
    ppg_tensor = torch.from_numpy(ppg_window.reshape(1, 1, -1)).float().to(device)

    model.eval()
    with torch.no_grad():
        pred = model(scal_tensor, ppg_tensor).cpu().numpy()

    sbp = sbp_scaler.inverse_transform(pred[:, 0:1])[0, 0]
    dbp = dbp_scaler.inverse_transform(pred[:, 1:2])[0, 0]
    return sbp, dbp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppg_csv', required=True)
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HybridBPModel(cfg).to(device)
    # Trigger lazy init
    dummy_scal = torch.zeros(1, 1, 128, 128).to(device)
    dummy_ppg = torch.zeros(1, 1, 1250).to(device)
    _ = model(dummy_scal, dummy_ppg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    sbp_scaler = joblib.load('checkpoints/sbp_scaler.pkl')
    dbp_scaler = joblib.load('checkpoints/dbp_scaler.pkl')

    df = pd.read_csv(args.ppg_csv)
    # Assume first numeric column is PPG
    ppg_raw = df.iloc[:, 0].values.astype(np.float32)

    fs = cfg['data']['sampling_rate']
    win = cfg['data']['window_size']
    ppg_proc = preprocess_ppg(ppg_raw, fs)

    results = []
    for start in range(0, len(ppg_proc) - win + 1, win // 2):
        window = ppg_proc[start:start + win]
        sbp, dbp = predict_from_window(model, window, cfg, device, sbp_scaler, dbp_scaler)
        if sbp is not None:
            t = start / fs
            results.append({'time_s': t, 'sbp': sbp, 'dbp': dbp})
            print(f"t={t:.1f}s → SBP: {sbp:.1f} mmHg | DBP: {dbp:.1f} mmHg")

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv('results/inference_output.csv', index=False)
        print(f"\nSaved {len(results)} predictions to results/inference_output.csv")


if __name__ == '__main__':
    main()
