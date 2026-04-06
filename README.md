#Noninvasive Blood Pressure Estimation Using a Hybrid Residual CNN-Transformer Model on PPG Signals"*
PyTorch implementation of:
> *"Noninvasive Blood Pressure Estimation Using a Hybrid Residual CNN-Transformer Model on PPG Signals"*

## Architecture
```
PPG (1250×1) ──► Conv1D ──► PosEnc ──► MultiHeadAttn(4h) ──► Dense(128) ─┐
                                                                            ├─► Concat ─► Dense(128) ─► Dropout ─► SBP/DBP
CWT Scalogram (128×128) ──► ResBlock(32) ──► ResBlock(64) ──► Dense(128) ─┘
```

## Results (paper — MIMIC-III)
| Metric | SBP | DBP |
|--------|-----|-----|
| MAE (mmHg) | 4.99 | 2.86 |
| R² | 0.812 | 0.844 |
| BHS Grade | approaches A | **A** |

## Quick Start
```bash
pip install -r requirements.txt
python test_pipeline.py                                         # smoke test
python src/generate_synthetic_data.py --n_files 50             # or use real MIMIC-III
python src/build_dataset.py --config configs/config.yaml       # build dataset
python main_train.py
                                       # train
python main_inference.py --ppg_csv your_ppg.csv                # inference
python src/ablation.py                                          # ablation study
python src/attention_viz.py --sample_id 0                      # attention maps
python src/export_onnx.py                                       # ONNX export
```

## Dataset
MIMIC-III Waveform DB: https://physionet.org/content/mimic3wdb/1.0/
Place CSV files (columns: ppg, abp @ 125Hz) in dataset/raw_csv/

## GPU Requirements
- Minimum: 6GB VRAM (batch=16)
- Recommended: 8GB VRAM (batch=32)
- Expected training time: ~2-4 hrs on RTX 3080
