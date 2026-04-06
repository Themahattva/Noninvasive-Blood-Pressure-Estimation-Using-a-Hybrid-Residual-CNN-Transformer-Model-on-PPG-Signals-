"""
Generates synthetic PPG + ABP CSV files that mimic MIMIC-III structure.
Use this to test the full pipeline without PhysioNet credentials.

Usage: python src/generate_synthetic_data.py --n_files 10 --out_dir dataset/raw_csv
"""
import os
import argparse
import numpy as np
import pandas as pd


def generate_ppg(duration_s=120, fs=125, hr_bpm=70, noise=0.05):
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    hr_hz = hr_bpm / 60.0
    # Realistic PPG: systolic peak + dicrotic notch
    phase = 2 * np.pi * hr_hz * t
    ppg = (
        0.6 * np.sin(phase)
        + 0.2 * np.sin(2 * phase)
        + 0.1 * np.sin(3 * phase)
        + 0.05 * np.cos(4 * phase)
    )
    # Add baseline wander (respiratory ~0.25 Hz)
    ppg += 0.3 * np.sin(2 * np.pi * 0.25 * t)
    # Shift to positive to match real PPG distribution and preprocess checks
    ppg += 1.0
    # Gaussian noise
    ppg += np.random.normal(0, noise, size=ppg.shape)
    return ppg.astype(np.float32)


def generate_abp(duration_s=120, fs=125, sbp=120, dbp=80, hr_bpm=70, noise=2.0):
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    hr_hz = hr_bpm / 60.0
    phase = 2 * np.pi * hr_hz * t
    map_val = dbp + (sbp - dbp) / 3.0
    amp = (sbp - dbp) / 2.0
    abp = map_val + amp * (
        np.sin(phase) + 0.15 * np.sin(2 * phase)
    )
    abp += np.random.normal(0, noise, size=abp.shape)
    abp = np.clip(abp, 40, 200)
    return abp.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_files', type=int, default=10)
    parser.add_argument('--out_dir', default='dataset/raw_csv')
    parser.add_argument('--duration_s', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.n_files):
        # Randomize subject physiology
        hr = np.random.randint(55, 100)
        sbp = np.random.randint(90, 170)
        dbp = np.random.randint(50, 100)
        dbp = min(dbp, sbp - 20)  # ensure pulse pressure >= 20

        ppg = generate_ppg(args.duration_s, 125, hr, noise=0.04 + 0.02 * np.random.rand())
        abp = generate_abp(args.duration_s, 125, sbp, dbp, hr, noise=1.5 + np.random.rand())

        df = pd.DataFrame({'ppg': ppg, 'abp': abp})
        fname = os.path.join(args.out_dir, f"subject_{i:04d}.csv")
        df.to_csv(fname, index=False)

    print(f"Generated {args.n_files} synthetic CSV files in '{args.out_dir}'")
    print("Each file: 2 columns (ppg, abp) | 125 Hz | ~300s")


if __name__ == '__main__':
    main()
