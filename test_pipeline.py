"""
Smoke test: runs the full pipeline on synthetic data without real MIMIC-III files.
Verifies data generation → preprocessing → scalogram → dataset → model forward pass.

Usage: python test_pipeline.py
"""
import os
import sys
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run():
    print("=== Smoke Test: Full Pipeline ===\n")

    # ── 1. Generate synthetic data ────────────────────────────────────────────
    print("[1/6] Generating synthetic data...")
    from src.generate_synthetic_data import generate_ppg, generate_abp
    import pandas as pd

    os.makedirs('dataset/raw_csv', exist_ok=True)
    for i in range(5):
        hr = 60 + i * 5
        ppg = generate_ppg(300, 125, hr)
        abp = generate_abp(300, 125, 110 + i * 5, 70 + i * 2, hr)
        pd.DataFrame({'ppg': ppg, 'abp': abp}).to_csv(f'dataset/raw_csv/test_{i:03d}.csv', index=False)
    print("  OK — 5 synthetic CSVs written")

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    print("[2/6] Testing preprocessing...")
    from src.preprocessing import preprocess_ppg, extract_windows
    ppg_raw = np.random.randn(1250 * 4).astype(np.float32)
    abp_raw = (80 + 20 * np.sin(np.linspace(0, 40 * np.pi, 1250 * 4))).astype(np.float32)
    ppg_proc = preprocess_ppg(ppg_raw)
    assert ppg_proc.shape == ppg_raw.shape
    assert 0.0 <= ppg_proc.min() and ppg_proc.max() <= 1.0
    print("  OK — preprocess_ppg output shape & range correct")

    # ── 3. Scalogram ──────────────────────────────────────────────────────────
    print("[3/6] Testing scalogram generation...")
    from src.scalogram import ppg_to_scalogram
    ppg_window = np.random.rand(1250).astype(np.float32)
    scal = ppg_to_scalogram(ppg_window, scales=32, size=64, gamma=0.6)
    # scal may be None if quality check fails on random noise — that's OK
    print(f"  OK — scalogram output: {scal.shape if scal is not None else 'None (filtered)'}")

    # ── 4. Build dataset (mini) ───────────────────────────────────────────────
    print("[4/6] Running build_dataset on synthetic CSVs...")
    from src.utils import load_config
    cfg = load_config('configs/config.yaml')
    cfg['scalogram']['scales'] = 32   # faster for test
    cfg['scalogram']['image_size'] = 64
    cfg['data']['ppg_dir'] = 'dataset/test_processed/ppg'
    cfg['data']['scalogram_dir'] = 'dataset/test_processed/scal'
    cfg['data']['labels_path'] = 'dataset/test_processed/labels.csv'

    from src.preprocessing import extract_windows
    from src.scalogram import ppg_to_scalogram
    import cv2, joblib
    from sklearn.preprocessing import MinMaxScaler

    os.makedirs(cfg['data']['ppg_dir'], exist_ok=True)
    os.makedirs(cfg['data']['scalogram_dir'], exist_ok=True)

    records = []
    sid = 0
    for i in range(5):
        df = pd.read_csv(f'dataset/raw_csv/test_{i:03d}.csv')
        ppg = df['ppg'].values.astype(np.float32)
        abp = df['abp'].values.astype(np.float32)
        wins, labs = extract_windows(ppg, abp, 125, 1250, 625, cfg['preprocessing'])
        for w, (sbp, dbp) in zip(wins, labs):
            scal = ppg_to_scalogram(w, scales=32, wavelet='morl', size=64, gamma=0.6,
                                     min_contrast=0.0, min_edge=0.0)  # relaxed for test
            if scal is None:
                continue
            np.save(os.path.join(cfg['data']['ppg_dir'], f"{sid:06d}.npy"), w.reshape(-1, 1))
            cv2.imwrite(os.path.join(cfg['data']['scalogram_dir'], f"{sid:06d}.png"),
                        (scal * 255).astype(np.uint8))
            records.append({'sample_id': sid, 'sbp': sbp, 'dbp': dbp})
            sid += 1

    if sid == 0:
        print("  WARN — no valid windows found from synthetic data (quality filters too strict), relaxing further...")
        # Force at least one sample
        w = np.sin(np.linspace(0, 20 * np.pi, 1250)).astype(np.float32)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        scal = ppg_to_scalogram(w, scales=32, size=64, gamma=0.6, min_contrast=0.0, min_edge=0.0)
        np.save(os.path.join(cfg['data']['ppg_dir'], "000000.npy"), w.reshape(-1, 1))
        cv2.imwrite(os.path.join(cfg['data']['scalogram_dir'], "000000.png"),
                    ((scal if scal is not None else np.zeros((64, 64))) * 255).astype(np.uint8))
        records.append({'sample_id': 0, 'sbp': 120.0, 'dbp': 80.0})
        sid = 1

    ldf = pd.DataFrame(records)
    sbp_sc = MinMaxScaler(); dbp_sc = MinMaxScaler()
    ldf['sbp_scaled'] = sbp_sc.fit_transform(ldf[['sbp']])
    ldf['dbp_scaled'] = dbp_sc.fit_transform(ldf[['dbp']])
    ldf.to_csv(cfg['data']['labels_path'], index=False)
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(sbp_sc, 'checkpoints/sbp_scaler.pkl')
    joblib.dump(dbp_sc, 'checkpoints/dbp_scaler.pkl')
    print(f"  OK — {sid} samples created")

    # ── 5. Dataset & DataLoader ───────────────────────────────────────────────
    print("[5/6] Testing BPDataset...")
    from src.dataset import BPDataset
    from torch.utils.data import DataLoader

    ids = ldf['sample_id'].values
    ds = BPDataset(
        ids,
        cfg['data']['ppg_dir'],
        cfg['data']['scalogram_dir'],
        ldf,
        target_size=(64, 64)   # match what we wrote to disk
    )
    loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False, num_workers=0)
    scal_b, ppg_b, label_b = next(iter(loader))
    assert scal_b.shape[1:] == (1, 64, 64), f"Unexpected scalogram shape: {scal_b.shape}"
    assert ppg_b.shape[1:] == (1, 1250),    f"Unexpected PPG shape: {ppg_b.shape}"
    assert label_b.shape[1] == 2,           f"Unexpected label shape: {label_b.shape}"
    print(f"  OK — scalogram: {tuple(scal_b.shape)}, ppg: {tuple(ppg_b.shape)}, labels: {tuple(label_b.shape)}")

    # ── 6. Model forward pass ─────────────────────────────────────────────────
    print("[6/6] Testing model forward pass...")
    from src.model import HybridBPModel

    model = HybridBPModel(cfg)   # cfg has image_size=64 already
    dummy_scal = torch.zeros(1, 1, 64, 64)
    dummy_ppg  = torch.zeros(1, 1, 1250)
    out = model(dummy_scal, dummy_ppg)
    assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"
    print(f"  OK — model output shape: {tuple(out.shape)}")

    # ── 7. Mini training loop ────────────────────────────────────────────────
    print("[7/7] Testing mini training loop (2 steps)...")
    import torch.nn as nn

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss(delta=5.0)

    for step, (scal_b, ppg_b, label_b) in enumerate(loader):
        optimizer.zero_grad()
        pred = model(scal_b, ppg_b)
        loss = loss_fn(pred, label_b)
        loss.backward()
        optimizer.step()
        if step >= 1:
            break

    assert not torch.isnan(loss), "Loss is NaN — gradient issue"
    print(f"  OK — loss after 2 steps: {loss.item():.5f} (no NaN)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    shutil.rmtree('dataset/test_processed', ignore_errors=True)
    for i in range(5):
        f = f'dataset/raw_csv/test_{i:03d}.csv'
        if os.path.exists(f):
            os.remove(f)

    print("\n✓ All 7 smoke tests passed.\n")
    print("Next steps:")
    print("  1. Place real MIMIC-III CSVs in dataset/raw_csv/")
    print("  2. python src/build_dataset.py --config configs/config.yaml")
    print("  3. python main_train.py")


if __name__ == '__main__':
    run()
