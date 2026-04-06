import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Scalogram Branch ────────────────────────────────────────────────────────

class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ScalogramBranch(nn.Module):
    def __init__(self, dropout=0.4, dense_units=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res1 = ResBlock2D(32, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = ResBlock2D(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        # After 3 MaxPool2d(2) on 128x128: 128/8=16 → 64*16*16=16384
        self.fc = nn.Sequential(
            nn.LazyLinear(dense_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.flatten(x)
        return self.fc(x)


# ── PPG Branch ───────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1250):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PPGBranch(nn.Module):
    def __init__(self, num_heads=4, key_dim=32, dense_units=128):
        super().__init__()
        d_model = num_heads * key_dim  # 128
        self.conv = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.proj = nn.Linear(64, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dense_units),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, 1, 1250)
        x = F.relu(self.conv(x))          # (B, 64, 1250)
        x = x.permute(0, 2, 1)            # (B, 1250, 64)
        x = self.proj(x)                  # (B, 1250, d_model)
        x = self.pos_enc(x)
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm2(x + attn_out)      # residual
        x = x.permute(0, 2, 1)            # (B, d_model, 1250)
        x = self.pool(x).squeeze(-1)      # (B, d_model)
        return self.fc(x)


# ── Hybrid Model ─────────────────────────────────────────────────────────────

class HybridBPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        self.cnn_branch = ScalogramBranch(m['dropout_cnn'], m['dense_units'])
        self.ppg_branch = PPGBranch(m['num_heads'], m['key_dim'], m['dense_units'])

        fused_in = m['dense_units'] * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, m['dense_units']),
            nn.ReLU(),
            nn.Dropout(m['dropout_fusion'])
        )
        self.out_sbp = nn.Linear(m['dense_units'], 1)
        self.out_dbp = nn.Linear(m['dense_units'], 1)

    def forward(self, scalogram, ppg):
        f_cnn = self.cnn_branch(scalogram)
        f_ppg = self.ppg_branch(ppg)
        fused = torch.cat([f_cnn, f_ppg], dim=1)
        fused = self.fusion(fused)
        sbp = self.out_sbp(fused)
        dbp = self.out_dbp(fused)
        return torch.cat([sbp, dbp], dim=1)  # (B, 2)
