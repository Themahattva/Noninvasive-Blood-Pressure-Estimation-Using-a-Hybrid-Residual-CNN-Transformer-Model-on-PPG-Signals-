import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2


class BPDataset(Dataset):
    def __init__(self, sample_ids, ppg_dir, scalogram_dir, labels_df, target_size=None):
        """
        target_size: (H, W) to resize scalograms on load. If None, uses as-is.
        Allows smoke tests with 64×64 and production runs with 128×128.
        """
        self.sample_ids = sample_ids
        self.ppg_dir = ppg_dir
        self.scal_dir = scalogram_dir
        self.labels = labels_df.set_index('sample_id')
        self.target_size = target_size  # (H, W) or None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sid_str = f"{sid:06d}"

        ppg = np.load(os.path.join(self.ppg_dir, f"{sid_str}.npy")).astype(np.float32)  # (T, 1)
        ppg = torch.from_numpy(ppg).permute(1, 0)  # (1, T)

        img = cv2.imread(os.path.join(self.scal_dir, f"{sid_str}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Scalogram not found: {sid_str}.png")
        if self.target_size is not None and (img.shape[0] != self.target_size[0] or img.shape[1] != self.target_size[1]):
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        scalogram = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        row = self.labels.loc[sid]
        sbp_s = float(row['sbp_scaled'])
        dbp_s = float(row['dbp_scaled'])
        label = torch.tensor([sbp_s, dbp_s], dtype=torch.float32)

        return scalogram, ppg, label


def get_dataloaders(cfg, seed=42):
    labels_df = pd.read_csv(cfg['data']['labels_path'])
    ppg_dir = cfg['data']['ppg_dir']
    scal_dir = cfg['data']['scalogram_dir']
    batch_size = cfg['training']['batch_size']
    img_size = cfg['scalogram']['image_size']
    target_size = (img_size, img_size)

    n = len(labels_df)
    if n < 10:
        raise ValueError(f"Dataset too small ({n} samples). Need at least 10.")

    # Stratified split on discretized SBP — reduce bins if dataset is small
    n_bins = min(10, max(2, n // 10))
    labels_df['sbp_bin'] = pd.cut(labels_df['sbp_scaled'], bins=n_bins, labels=False)
    # Fill any NaN bins (edge cases) with 0
    labels_df['sbp_bin'] = labels_df['sbp_bin'].fillna(0).astype(int)

    all_ids = labels_df['sample_id'].values
    bins = labels_df['sbp_bin'].values

    # Only stratify if enough samples per bin; else fall back to random split
    bin_counts = np.bincount(bins)
    use_stratify = bool(bin_counts.min() >= 2)

    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.30, random_state=seed,
        stratify=bins if use_stratify else None
    )
    temp_labels = labels_df[labels_df['sample_id'].isin(temp_ids)]
    temp_bins = temp_labels['sbp_bin'].values
    temp_bin_counts = np.bincount(temp_bins, minlength=n_bins)
    use_stratify_temp = bool(temp_bin_counts.min() >= 2)

    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=seed,
        stratify=temp_bins if use_stratify_temp else None
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_ds = BPDataset(train_ids, ppg_dir, scal_dir, labels_df, target_size)
    val_ds   = BPDataset(val_ids,   ppg_dir, scal_dir, labels_df, target_size)
    test_ds  = BPDataset(test_ids,  ppg_dir, scal_dir, labels_df, target_size)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=min(batch_size, len(val_ds)),
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=min(batch_size, len(test_ds)),
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
