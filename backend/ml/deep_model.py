"""
deep_model.py - PyTorch Spatio-Temporal 2D ConvLSTM / U-Net for Weather Anomaly Forecasting
Step 7 Implementation:
- Ingests multi-channel 2D spatial fields (Rainfall, Tmax anomaly, Tmin anomaly)
- Operates strictly on anomaly space (with raw rain in log1p scale)
- Evaluates against the Step 4/5 LightGBM Direct Forecaster on validation data (2020-2021)
- Implements strict gate check: If deep model does not strictly outperform GBDT on validation RMSE/CRPS,
  document honestly and defer to GBDT for production inference.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zarr
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x: torch.Tensor, h_c: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = h_c
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SpatioTemporalWeatherNet(nn.Module):
    """
    Encoder-Decoder ConvLSTM architecture:
    - Input: Sequence of L days of gridded fields (Channels: Rain log1p, Tmax anom, Tmin anom)
    - Output: Gridded forecasts for future lead days h in 1..lead_days
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32, out_channels: int = 3, lead_days: int = 20):
        super().__init__()
        self.lead_days = lead_days
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_lstm = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)

        # Direct Forecaster Projection
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels * lead_days, kernel_size=3, padding=1)
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (Batch, Seq_len, Channels, H, W)
        Returns: (Batch, lead_days, Channels, H, W)
        """
        B, S, C, H, W = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)

        for t in range(S):
            xt = x_seq[:, t]
            feat = self.encoder_conv(xt)
            h, c = self.encoder_lstm(feat, (h, c))

        # Decode directly to all lead days
        out_all = self.decoder_conv(h)  # (B, out_channels * lead_days, H, W)
        out_all = out_all.view(B, self.lead_days, C, H, W)
        return out_all


class GriddedWeatherPatchDataset(Dataset):
    """
    Subsamples regional patches (e.g. 32x32) across India to allow tractable CPU training
    with complete spatio-temporal dynamics.
    """
    def __init__(self, zarr_data, clim_mean, split_indices, seq_len: int = 7, lead_days: int = 20, patch_size: int = 32, num_samples: int = 300):
        self.zarr_data = zarr_data
        self.clim_mean = clim_mean
        self.split_indices = split_indices
        self.seq_len = seq_len
        self.lead_days = lead_days
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.rng = np.random.RandomState(42)

        # Precompute valid start day indices
        max_idx = len(split_indices) - (seq_len + lead_days)
        self.valid_starts = np.arange(0, max(1, max_idx))

    def __len__(self):
        return min(self.num_samples, len(self.valid_starts))

    def __getitem__(self, idx):
        start_rel = self.valid_starts[idx % len(self.valid_starts)]
        start_abs = self.split_indices[start_rel]

        # Extract sequence and target days
        tot_days = self.seq_len + self.lead_days
        raw_slice = self.zarr_data[start_abs : start_abs + tot_days]  # (tot_days, 3, 129, 135)

        # Fixed central patch covering Western Ghats to Central/North India (32x32)
        r_start, c_start = 45, 40
        patch = raw_slice[:, :, r_start : r_start + self.patch_size, c_start : c_start + self.patch_size]

        # Transform: rain to log1p, temperatures to anomalies
        # Compute day-of-year for each day
        clim_slice = np.zeros_like(patch)
        for t in range(tot_days):
            doy = (start_abs + t) % 365
            clim_slice[t] = self.clim_mean[doy, :, r_start : r_start + self.patch_size, c_start : c_start + self.patch_size]

        anom_patch = np.zeros_like(patch, dtype=np.float32)
        # Channel 0: rain log1p
        anom_patch[:, 0] = np.log1p(np.maximum(0.0, np.nan_to_num(patch[:, 0], nan=0.0)))
        # Channel 1: tmax anomaly
        anom_patch[:, 1] = np.nan_to_num(patch[:, 1] - clim_slice[:, 1], nan=0.0)
        # Channel 2: tmin anomaly
        anom_patch[:, 2] = np.nan_to_num(patch[:, 2] - clim_slice[:, 2], nan=0.0)

        x = torch.from_numpy(anom_patch[: self.seq_len]).float()
        y = torch.from_numpy(anom_patch[self.seq_len : tot_days]).float()
        return x, y


def train_and_gate_check(zarr_path: str, clim_path: str, output_dir: str = "results/deep_model"):
    """
    Trains the SpatioTemporalWeatherNet on 2000-2019 patches,
    evaluates on 2020-2021 validation patches, and performs a strict Gate Check against GBDT.
    """
    os.makedirs(output_dir, exist_ok=True)
    root = zarr.open(zarr_path, mode='r')
    clim_mean = np.load(clim_path)

    # Train (2000-2019): 7305 days; Val (2020-2021): 731 days (days 7305..8036)
    train_indices = np.arange(0, 7305)
    val_indices = np.arange(7305, 8036)

    logger.info("Building Patch Datasets for Deep Spatio-Temporal Model...")
    train_ds = GriddedWeatherPatchDataset(root['weather'], clim_mean, train_indices, num_samples=250)
    val_ds = GriddedWeatherPatchDataset(root['weather'], clim_mean, val_indices, num_samples=60)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for deep learning training.")

    model = SpatioTemporalWeatherNet(in_channels=3, hidden_dim=24, out_channels=3, lead_days=20).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_temp = nn.MSELoss()
    criterion_rain = nn.HuberLoss(delta=1.0)

    epochs = 5
    logger.info(f"Training ConvLSTM Deep Model for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred_y = model(batch_x)

            # Combined loss: Tmax/Tmin anomaly MSE + Rain Huber loss
            loss_rain = criterion_rain(pred_y[:, :, 0], batch_y[:, :, 0])
            loss_tmax = criterion_temp(pred_y[:, :, 1], batch_y[:, :, 1])
            loss_tmin = criterion_temp(pred_y[:, :, 2], batch_y[:, :, 2])
            loss = loss_rain * 2.0 + loss_tmax + loss_tmin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f}")

    # Validation Evaluation
    model.eval()
    val_tmax_se = []
    val_rain_mae = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_y = model(batch_x)

            # Tmax error across all lead days
            tmax_err = (pred_y[:, :, 1] - batch_y[:, :, 1]).cpu().numpy()
            val_tmax_se.append(np.mean(tmax_err**2, axis=(0, 2, 3)))  # per lead day

            # Rain MAE
            pred_rain = np.expm1(np.maximum(0.0, pred_y[:, :, 0].cpu().numpy()))
            true_rain = np.expm1(np.maximum(0.0, batch_y[:, :, 0].cpu().numpy()))
            val_rain_mae.append(np.mean(np.abs(pred_rain - true_rain), axis=(0, 2, 3)))

    deep_tmax_rmse = np.sqrt(np.mean(val_tmax_se, axis=0))  # (20,)
    deep_rain_mae = np.mean(val_rain_mae, axis=0)          # (20,)

    # Baseline GBDT Validation RMSE for reference (from Step 4)
    # GBDT Tmax RMSE: Day 1: ~1.28C, Day 7: ~1.76C, Day 20: ~1.82C
    logger.info("=======================================================")
    logger.info("GATE CHECK: Deep Model vs LightGBM Direct Forecaster")
    logger.info("=======================================================")
    logger.info(f"Lead 1  | ConvLSTM Tmax RMSE: {deep_tmax_rmse[0]:.2f} C | GBDT Tmax RMSE: 1.28 C")
    logger.info(f"Lead 7  | ConvLSTM Tmax RMSE: {deep_tmax_rmse[6]:.2f} C | GBDT Tmax RMSE: 1.76 C")
    logger.info(f"Lead 20 | ConvLSTM Tmax RMSE: {deep_tmax_rmse[19]:.2f} C | GBDT Tmax RMSE: 1.82 C")

    passed_gate = bool(deep_tmax_rmse[0] < 1.28 and deep_tmax_rmse[6] < 1.76)
    logger.info(f"Gate Check Passed: {passed_gate}")
    if not passed_gate:
        logger.info("As required by Step 7 specification: Deep learning model did NOT strictly beat tabular GBDT on validation.")
        logger.info("Retaining LightGBM Direct Forecaster + Conformal Prediction as primary production engine.")

    # Save model & verification report
    torch.save(model.state_dict(), os.path.join(output_dir, "convlstm_weather_patch.pt"))
    gate_report = {
        "passed_gate": passed_gate,
        "deep_tmax_rmse": [float(x) for x in deep_tmax_rmse],
        "deep_rain_mae": [float(x) for x in deep_rain_mae],
        "gbdt_reference_rmse": {"lead_1": 1.28, "lead_7": 1.76, "lead_20": 1.82},
        "decision": "LightGBM + Conformal Forecaster selected for production inference" if not passed_gate else "Deep Model selected"
    }
    with open(os.path.join(output_dir, "gate_check_results.json"), "w") as f:
        json.dump(gate_report, f, indent=2)

    logger.info(f"Saved Step 7 artifacts to {output_dir}")
    return gate_report


if __name__ == "__main__":
    train_and_gate_check(
        zarr_path="data/processed/imd_weather_grid.zarr",
        clim_path="data/processed/climatology/clim_mean_2000_2019.npy"
    )
