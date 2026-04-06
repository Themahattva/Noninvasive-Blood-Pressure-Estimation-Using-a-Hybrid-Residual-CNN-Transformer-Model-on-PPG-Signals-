"""
Exports the trained HybridBPModel to ONNX format for edge/mobile deployment.

Usage: python src/export_onnx.py --checkpoint checkpoints/best_model.pt
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import HybridBPModel
from src.utils import load_config


def export_onnx(cfg, checkpoint_path, out_path='checkpoints/bp_model.onnx', opset=14):
    device = torch.device('cpu')
    model = HybridBPModel(cfg).to(device)

    # Trigger lazy init
    dummy_scal = torch.zeros(1, 1, 128, 128)
    dummy_ppg = torch.zeros(1, 1, 1250)
    _ = model(dummy_scal, dummy_ppg)

    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_scal, dummy_ppg),
        out_path,
        input_names=['scalogram', 'ppg'],
        output_names=['bp_prediction'],
        dynamic_axes={
            'scalogram': {0: 'batch'},
            'ppg': {0: 'batch'},
            'bp_prediction': {0: 'batch'}
        },
        opset_version=opset
    )
    print(f"Model exported to ONNX: {out_path}")
    print(f"Inputs:  scalogram (B,1,128,128), ppg (B,1,1250)")
    print(f"Outputs: bp_prediction (B,2) — [sbp_scaled, dbp_scaled]")
    print("Note: inverse-transform outputs with checkpoints/sbp_scaler.pkl / dbp_scaler.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--out', default='checkpoints/bp_model.onnx')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    export_onnx(cfg, args.checkpoint, args.out)


if __name__ == '__main__':
    main()
