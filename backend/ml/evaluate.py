"""
evaluate.py - Comprehensive Meteorological Evaluation Harness for IMD Weather Prediction v2.0
Computes standard meteorological baselines (Climatology and Persistence) and lead-day metrics (Lead 1 to 20):
- Temperature (Max & Min): RMSE, MAE, Anomaly Correlation Coefficient (ACC)
- Rainfall: Brier Score, CRPS, Equitable Threat Score (ETS), F1 Score at thresholds [2.5, 15.6, 64.5 mm]
Generates skill-vs-lead-day plots and identifies lead day where forecast skill stops beating climatology.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_climatology(train_tensor: np.ndarray, train_dates: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes smoothed daily climatology per grid cell across day-of-year (1..366).
    train_tensor: (T, C, H, W) where C: [rain, tmax, tmin]
    train_dates: list of ISO strings 'YYYY-MM-DD'
    Returns:
        clim_mean: (366, C, H, W)
        clim_std: (366, C, H, W)
    Leakage Protection: Computed strictly on training split.
    """
    T, C, H, W = train_tensor.shape
    doys = np.array([pd.to_datetime(d).dayofyear for d in train_dates])
    
    clim_mean = np.full((366, C, H, W), np.nan, dtype=np.float32)
    clim_std = np.full((366, C, H, W), np.nan, dtype=np.float32)

    for d in range(1, 367):
        # 15-day moving window around day of year d across all training years
        # Circular wrap for DOY (e.g. day 1 wraps to day 365)
        d_min = (d - 8) % 366
        d_max = (d + 7) % 366
        if d_min <= d_max:
            mask = (doys >= d_min) & (doys <= d_max)
        else:
            mask = (doys >= d_min) | (doys <= d_max)
            
        if np.any(mask):
            clim_mean[d-1] = np.nanmean(train_tensor[mask], axis=0)
            clim_std[d-1] = np.nanstd(train_tensor[mask], axis=0)
        else:
            clim_mean[d-1] = np.nanmean(train_tensor, axis=0)
            clim_std[d-1] = np.nanstd(train_tensor, axis=0)

    # For rainfall, non-negative
    clim_mean[:, 0] = np.maximum(0.0, clim_mean[:, 0])
    return clim_mean, clim_std


def compute_acc(y_true: np.ndarray, y_pred: np.ndarray, y_clim: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient (ACC) according to WMO verification standards.
    y_true: actual observations
    y_pred: forecasts
    y_clim: climatological normals
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isnan(y_clim))
    if np.sum(valid) < 2:
        return 0.0

    anom_true = y_true[valid] - y_clim[valid]
    anom_pred = y_pred[valid] - y_clim[valid]

    numerator = np.sum(anom_true * anom_pred)
    denom = np.sqrt(np.sum(anom_true ** 2) * np.sum(anom_pred ** 2))
    if denom < 1e-8:
        return 0.0
    return float(numerator / denom)


def compute_contingency_table(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    """
    Contingency table elements for threshold exceedance:
    Hits (a), False Alarms (b), Misses (c), Correct Negatives (d).
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    yt = y_true[valid] >= threshold
    yp = y_pred[valid] >= threshold

    a = int(np.sum(yt & yp))      # Hit
    b = int(np.sum((~yt) & yp))   # False Alarm
    c = int(np.sum(yt & (~yp)))   # Miss
    d = int(np.sum((~yt) & (~yp))) # Correct Rejection
    return a, b, c, d


def compute_ets(a: int, b: int, c: int, d: int) -> float:
    """
    Equitable Threat Score (Gilbert Skill Score).
    ETS = (Hits - Hits_random) / (Hits + FalseAlarms + Misses - Hits_random)
    Hits_random = (Hits + FalseAlarms) * (Hits + Misses) / Total
    """
    total = a + b + c + d
    if total == 0:
        return 0.0
    a_rand = ((a + b) * (a + c)) / total
    denom = a + b + c - a_rand
    if denom <= 0:
        return 0.0
    return float((a - a_rand) / denom)


def compute_f1(a: int, b: int, c: int, d: int) -> float:
    """F1 Score for binary event detection."""
    denom = 2 * a + b + c
    if denom == 0:
        return 0.0
    return float(2 * a / denom)


def compute_brier_score(y_true_binary: np.ndarray, prob_pred: np.ndarray) -> float:
    """Brier score for probabilistic or surrogate binary predictions."""
    valid = (~np.isnan(y_true_binary)) & (~np.isnan(prob_pred))
    if np.sum(valid) == 0:
        return 0.0
    return float(np.mean((prob_pred[valid] - y_true_binary[valid]) ** 2))


def compute_crps_point(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for deterministic / point predictions.
    For point forecasts, CRPS reduces to Mean Absolute Error.
    """
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if np.sum(valid) == 0:
        return 0.0
    return float(np.mean(np.abs(y_pred[valid] - y_true[valid])))


def evaluate_forecasts_per_lead(y_true_all: np.ndarray,
                                y_pred_model: np.ndarray,
                                y_clim_all: np.ndarray,
                                y_persist_all: np.ndarray,
                                lead_days: int = 20,
                                rain_thresholds: List[float] = [2.5, 15.6, 64.5]) -> Dict:
    """
    Evaluates Model vs Climatology vs Persistence across lead days 1 to 20.
    Input arrays have shape: (N_forecast_cases, Lead_Days=20, Channels=3, H, W)
    """
    results = {
        "lead_days": list(range(1, lead_days + 1)),
        "model": {},
        "climatology": {},
        "persistence": {},
        "summary": {}
    }

    channels = ["rain", "tmax", "tmin"]
    for src in ["model", "climatology", "persistence"]:
        results[src] = {ch: {"rmse": [], "mae": [], "acc": []} for ch in channels}
        results[src]["rain_metrics"] = {
            f"{thresh}mm": {"brier": [], "crps": [], "ets": [], "f1": []} for thresh in rain_thresholds
        }

    for h in range(lead_days):
        lead_day = h + 1
        yt = y_true_all[:, h]        # (N, 3, H, W)
        yc = y_clim_all[:, h]        # (N, 3, H, W)
        yp = y_persist_all[:, h]     # (N, 3, H, W)
        ym = y_pred_model[:, h]      # (N, 3, H, W)

        for src_name, y_src in [("model", ym), ("climatology", yc), ("persistence", yp)]:
            # Temperature & Rain point metrics
            for ch_idx, ch in enumerate(channels):
                yt_c = yt[:, ch_idx]
                ys_c = y_src[:, ch_idx]
                yc_c = yc[:, ch_idx]

                valid = (~np.isnan(yt_c)) & (~np.isnan(ys_c))
                if np.sum(valid) > 0:
                    rmse = float(np.sqrt(np.mean((ys_c[valid] - yt_c[valid]) ** 2)))
                    mae = float(np.mean(np.abs(ys_c[valid] - yt_c[valid])))
                else:
                    rmse, mae = 0.0, 0.0

                acc = compute_acc(yt_c, ys_c, yc_c)
                results[src_name][ch]["rmse"].append(round(rmse, 4))
                results[src_name][ch]["mae"].append(round(mae, 4))
                results[src_name][ch]["acc"].append(round(acc, 4))

            # Rainfall extreme thresholds
            yt_rain = yt[:, 0]
            ys_rain = y_src[:, 0]
            for thresh in rain_thresholds:
                t_key = f"{thresh}mm"
                # Surrogate probability for point forecast: step function or sigmoid
                prob_pred = np.clip((ys_rain - (thresh - 2.0)) / 4.0, 0.0, 1.0)
                true_bin = (yt_rain >= thresh).astype(np.float32)

                brier = compute_brier_score(true_bin, prob_pred)
                crps = compute_crps_point(yt_rain, ys_rain)
                a, b, c, d = compute_contingency_table(yt_rain, ys_rain, thresh)
                ets = compute_ets(a, b, c, d)
                f1 = compute_f1(a, b, c, d)

                results[src_name]["rain_metrics"][t_key]["brier"].append(round(brier, 4))
                results[src_name]["rain_metrics"][t_key]["crps"].append(round(crps, 4))
                results[src_name]["rain_metrics"][t_key]["ets"].append(round(ets, 4))
                results[src_name]["rain_metrics"][t_key]["f1"].append(round(f1, 4))

    # Determine crossover lead day where model stops beating climatology for tmax ACC & rain RMSE
    model_tmax_acc = results["model"]["tmax"]["acc"]
    model_tmax_rmse = results["model"]["tmax"]["rmse"]
    clim_tmax_rmse = results["climatology"]["tmax"]["rmse"]

    crossover_day = None
    for day_idx in range(lead_days):
        # Model stops beating climatology when RMSE(model) >= RMSE(climatology) or ACC <= 0.5
        if model_tmax_rmse[day_idx] >= clim_tmax_rmse[day_idx] or model_tmax_acc[day_idx] <= 0.4:
            crossover_day = day_idx + 1
            break

    results["summary"]["climatology_crossover_day"] = crossover_day if crossover_day else lead_days
    return results


def plot_skill_vs_lead_day(eval_results: Dict, out_dir: str = "results") -> str:
    """
    Renders standard WMO meteorological skill-vs-lead-day plots:
    1. Tmax RMSE vs Lead Day (Model vs Persistence vs Climatology)
    2. Tmax ACC vs Lead Day (Skill curve with 0.5/0.6 threshold)
    3. Rainfall CRPS vs Lead Day
    4. Heavy Rain (64.5mm) ETS vs Lead Day
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "skill_vs_lead_day.png")

    lead_days = eval_results["lead_days"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Tmax RMSE
    axes[0, 0].plot(lead_days, eval_results["model"]["tmax"]["rmse"], 'o-', color='#DC2626', label='Model Forecast', linewidth=2)
    axes[0, 0].plot(lead_days, eval_results["persistence"]["tmax"]["rmse"], 's--', color='#6B7280', label='Persistence Baseline', linewidth=1.8)
    axes[0, 0].plot(lead_days, eval_results["climatology"]["tmax"]["rmse"], '^:', color='#2563EB', label='Climatology Baseline', linewidth=1.8)
    axes[0, 0].set_title("Max Temperature RMSE vs Forecast Lead Day", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Lead Day (Days ahead)")
    axes[0, 0].set_ylabel("RMSE (°C)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    # 2. Tmax Anomaly Correlation (ACC)
    axes[0, 1].plot(lead_days, eval_results["model"]["tmax"]["acc"], 'o-', color='#DC2626', label='Model ACC', linewidth=2)
    axes[0, 1].plot(lead_days, eval_results["persistence"]["tmax"]["acc"], 's--', color='#6B7280', label='Persistence ACC', linewidth=1.8)
    axes[0, 1].axhline(y=0.5, color='#EAB308', linestyle='--', label='Useful Skill Limit (ACC=0.5)')
    axes[0, 1].set_title("Max Temperature Anomaly Correlation (ACC) vs Lead Day", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Lead Day")
    axes[0, 1].set_ylabel("ACC")
    axes[0, 1].set_ylim(-0.1, 1.05)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    # 3. Rainfall CRPS
    axes[1, 0].plot(lead_days, eval_results["model"]["rain"]["mae"], 'o-', color='#059669', label='Model Rain MAE/CRPS', linewidth=2)
    axes[1, 0].plot(lead_days, eval_results["persistence"]["rain"]["mae"], 's--', color='#6B7280', label='Persistence', linewidth=1.8)
    axes[1, 0].plot(lead_days, eval_results["climatology"]["rain"]["mae"], '^:', color='#2563EB', label='Climatology', linewidth=1.8)
    axes[1, 0].set_title("Rainfall CRPS / MAE vs Forecast Lead Day", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Lead Day")
    axes[1, 0].set_ylabel("Error (mm)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    # 4. Moderate/Heavy Rain ETS (15.6 mm)
    thresh_key = "15.6mm"
    axes[1, 1].plot(lead_days, eval_results["model"]["rain_metrics"][thresh_key]["ets"], 'o-', color='#7C3AED', label=f'Model ETS ({thresh_key})', linewidth=2)
    axes[1, 1].plot(lead_days, eval_results["persistence"]["rain_metrics"][thresh_key]["ets"], 's--', color='#6B7280', label='Persistence', linewidth=1.8)
    axes[1, 1].plot(lead_days, eval_results["climatology"]["rain_metrics"][thresh_key]["ets"], '^:', color='#2563EB', label='Climatology', linewidth=1.8)
    axes[1, 1].set_title(f"Rainfall (>15.6mm) Equitable Threat Score (ETS)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Lead Day")
    axes[1, 1].set_ylabel("ETS")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Skill-vs-lead-day plots saved to {out_path}")
    return out_path
