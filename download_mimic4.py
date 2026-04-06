import wfdb
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

OUT_DIR = "dataset/raw_csv"
os.makedirs(OUT_DIR, exist_ok=True)

print("Fetching record list from MIMIC-IV Waveform DB...")
all_records = wfdb.get_record_list('mimic4wdb/0.1.0')
print(f"Found {len(all_records)} records")

saved = 0
for rec_path in tqdm(all_records, desc="Downloading"):
    try:
        record = wfdb.rdrecord(rec_path, pn_dir='mimic4wdb/0.1.0')
        sig_names_lower = [s.lower() for s in record.sig_name]

        ppg_idx = next((i for i, s in enumerate(sig_names_lower)
                        if 'pleth' in s or 'ppg' in s), None)
        abp_idx = next((i for i, s in enumerate(sig_names_lower)
                        if 'abp' in s or 'art' in s), None)

        if ppg_idx is None or abp_idx is None:
            continue

        ppg = record.p_signal[:, ppg_idx].astype(np.float32)
        abp = record.p_signal[:, abp_idx].astype(np.float32)

        # Drop NaN
        mask = ~(np.isnan(ppg) | np.isnan(abp))
        ppg, abp = ppg[mask], abp[mask]

        if len(ppg) < 1250:
            continue

        fname = rec_path.replace('/', '_').strip('_') + '.csv'
        pd.DataFrame({'ppg': ppg, 'abp': abp}).to_csv(
            os.path.join(OUT_DIR, fname), index=False
        )
        saved += 1
        print(f"  Saved [{saved}]: {fname}  ({len(ppg)} samples)")

    except Exception as e:
        continue

print(f"\nDone. {saved} waveform CSV files saved to '{OUT_DIR}'")