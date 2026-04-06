"""
Usage: python build_dataset.py --config configs/config.yaml
Expects CSV files with columns: ppg, abp (or time, ppg, abp)
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
import yaml
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import extract_windows
from src.scalogram import ppg_to_scalogram


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_ppg_abp_columns(df):
    cols = [c.lower() for c in df.columns]
    ppg_col = next((df.columns[i] for i, c in enumerate(cols) if 'ppg' in c or 'pleth' in c), None)
    abp_col = next((df.columns[i] for i, c in enumerate(cols) if 'abp' in c or 'art' in c or 'bp' in c), None)
    if ppg_col is None or abp_col is None:
        # fallback: assume col 0 = ppg, col 1 = abp (or col 1,2 if time present)
        if len(df.columns) >= 3:
            ppg_col, abp_col = df.columns[1], df.columns[2]
        else:
            ppg_col, abp_col = df.columns[0], df.columns[1]
    return ppg_col, abp_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_dir = cfg['data']['raw_csv_dir']
    ppg_dir = cfg['data']['ppg_dir']
    scal_dir = cfg['data']['scalogram_dir']
    labels_path = cfg['data']['labels_path']
    os.makedirs(ppg_dir, exist_ok=True)
    os.makedirs(scal_dir, exist_ok=True)

    fs = cfg['data']['sampling_rate']
    win = cfg['data']['window_size']
    overlap = cfg['data']['overlap']
    pre_cfg = cfg['preprocessing']
    scal_cfg = cfg['scalogram']

    records = []
    sample_id = 0
    csv_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])

    for fname in tqdm(csv_files, desc="Processing CSV files"):
        fpath = os.path.join(raw_dir, fname)
        try:
            df = pd.read_csv(fpath)
            ppg_col, abp_col = find_ppg_abp_columns(df)
            ppg = df[ppg_col].values.astype(np.float32)
            abp = df[abp_col].values.astype(np.float32)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        ppg_wins, labels = extract_windows(ppg, abp, fs, win, overlap, pre_cfg)

        for ppg_w, (sbp, dbp) in zip(ppg_wins, labels):
            scalogram = ppg_to_scalogram(
                ppg_w,
                scales=scal_cfg['scales'],
                wavelet=scal_cfg['wavelet'],
                size=scal_cfg['image_size'],
                gamma=scal_cfg['gamma']
            )
            if scalogram is None:
                continue

            npy_path = os.path.join(ppg_dir, f"{sample_id:06d}.npy")
            png_path = os.path.join(scal_dir, f"{sample_id:06d}.png")

            np.save(npy_path, ppg_w.reshape(-1, 1).astype(np.float32))
            cv2.imwrite(png_path, (scalogram * 255).astype(np.uint8))

            records.append({'sample_id': sample_id, 'sbp': sbp, 'dbp': dbp})
            sample_id += 1

    labels_df = pd.DataFrame(records)

    if labels_df.empty:
        raise ValueError(
            f"No valid windows found in raw CSVs at '{raw_dir}'. "
            "Check that source CSV files contain PPG/ABP data and window settings are correct."
        )

    for col in ['sbp', 'dbp']:
        if col not in labels_df.columns:
            raise ValueError(f"Expected label column '{col}' missing from generated records: {labels_df.columns.tolist()}")

    # Fit and save BP scalers
    sbp_scaler = MinMaxScaler()
    dbp_scaler = MinMaxScaler()
    labels_df['sbp_scaled'] = sbp_scaler.fit_transform(labels_df[['sbp']])
    labels_df['dbp_scaled'] = dbp_scaler.fit_transform(labels_df[['dbp']])
    labels_df.to_csv(labels_path, index=False)

    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(sbp_scaler, 'checkpoints/sbp_scaler.pkl')
    joblib.dump(dbp_scaler, 'checkpoints/dbp_scaler.pkl')

    print(f"Total valid samples: {sample_id}")
    print(f"Labels saved to: {labels_path}")


if __name__ == '__main__':
    main()
