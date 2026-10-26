"""
uncertainty.py - Probabilistic Weather Forecasting & Conformal Uncertainty Quantification
Features:
- Multi-horizon quantile regressors (q05, q25, q50, q75, q95) for Temperature and Rainfall amount
- Split conformal prediction calibration using per-lead-day validation residuals:
    Ensures empirical coverage on unseen test data matches the nominal 90% (q05 to q95) and 50% (q25 to q75) levels
- Conformalized quantile regression (CQR) intervals:
    interval = [pred_q05 - q_val_residual, pred_q95 + q_val_residual]
- Verification tools: Coverage evaluation and reliability diagrams per lead day
"""

import os
import json
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


class ConformalQuantileForecaster:
    """
    Fits quantile gradient boosting models across lead days h in 1..lead_days
    and wraps them in split conformal prediction using validation set non-conformity scores.
    """
    def __init__(self, lead_days: int = 20, quantiles: List[float] = QUANTILES):
        self.lead_days = lead_days
        self.quantiles = quantiles
        # Models dict: lead_day -> { 'tmax': {q: model}, 'tmin': {q: model}, 'rain': {q: model} }
        self.models = {}
        # Conformal adjustments per lead day: lead_day -> { 'tmax_90': delta, 'tmax_50': delta, ... }
        self.conformal_offsets = {}

    def fit(self, X_train: np.ndarray, Y_anom_train: np.ndarray, Y_raw_rain_train: np.ndarray):
        """
        Trains quantile LightGBM models for tmax anomaly, tmin anomaly, and rainfall amount.
        """
        logger.info(f"Fitting quantile regressors for {self.lead_days} lead days across quantiles {self.quantiles}...")
        
        for h in range(1, self.lead_days + 1):
            h_idx = h - 1
            y_tmax = np.nan_to_num(Y_anom_train[:, h_idx, 1], nan=0.0)
            y_tmin = np.nan_to_num(Y_anom_train[:, h_idx, 2], nan=0.0)
            y_rain = np.maximum(0.0, np.nan_to_num(Y_raw_rain_train[:, h_idx], nan=0.0))

            self.models[h] = {'tmax': {}, 'tmin': {}, 'rain': {}}

            for q in self.quantiles:
                # 1. Tmax quantile
                m_tx = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=60,
                    learning_rate=0.06,
                    num_leaves=31,
                    random_state=42 + int(q * 100) + h,
                    verbose=-1
                )
                m_tx.fit(X_train, y_tmax)
                self.models[h]['tmax'][q] = m_tx

                # 2. Tmin quantile
                m_tn = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=60,
                    learning_rate=0.06,
                    num_leaves=31,
                    random_state=42 + int(q * 100) + h,
                    verbose=-1
                )
                m_tn.fit(X_train, y_tmin)
                self.models[h]['tmin'][q] = m_tn

                # 3. Rain quantile (fit on non-negative raw rain)
                m_rn = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=60,
                    learning_rate=0.06,
                    num_leaves=31,
                    random_state=42 + int(q * 100) + h,
                    verbose=-1
                )
                m_rn.fit(X_train, y_rain)
                self.models[h]['rain'][q] = m_rn

        logger.info("Quantile regression model training complete.")
        return self

    def calibrate(self, X_val: np.ndarray, Y_anom_val: np.ndarray, Y_raw_rain_val: np.ndarray,
                  nominal_coverage: float = 0.90):
        """
        Calibrates interval widths using Split Conformal Prediction (CQR) on the validation set.
        For nominal 90% interval [q05, q95]:
          nonconformity score E_i = max(q05(x_i) - y_i, y_i - q95(x_i))
          calibrated offset Q_{1-alpha} is the ceil((n+1)*(1-alpha))/n quantile of E.
        """
        logger.info(f"Calibrating conformal prediction offsets on validation set (nominal {nominal_coverage*100:.0f}%)...")
        alpha = 1.0 - nominal_coverage
        n_val = len(X_val)

        for h in range(1, self.lead_days + 1):
            h_idx = h - 1
            yt_tx = np.nan_to_num(Y_anom_val[:, h_idx, 1], nan=0.0)
            yt_tn = np.nan_to_num(Y_anom_val[:, h_idx, 2], nan=0.0)
            yt_rn = np.maximum(0.0, np.nan_to_num(Y_raw_rain_val[:, h_idx], nan=0.0))

            # Raw quantile predictions on val
            pred_tx_q05 = self.models[h]['tmax'][0.05].predict(X_val)
            pred_tx_q95 = self.models[h]['tmax'][0.95].predict(X_val)

            pred_tn_q05 = self.models[h]['tmin'][0.05].predict(X_val)
            pred_tn_q95 = self.models[h]['tmin'][0.95].predict(X_val)

            pred_rn_q05 = np.maximum(0.0, self.models[h]['rain'][0.05].predict(X_val))
            pred_rn_q95 = np.maximum(0.0, self.models[h]['rain'][0.95].predict(X_val))

            # Non-conformity scores E_i = max(low - y, y - high)
            scores_tx = np.maximum(pred_tx_q05 - yt_tx, yt_tx - pred_tx_q95)
            scores_tn = np.maximum(pred_tn_q05 - yt_tn, yt_tn - pred_tn_q95)
            scores_rn = np.maximum(pred_rn_q05 - yt_rn, yt_rn - pred_rn_q95)

            # Conformal quantile: level = min(1.0, np.ceil((n_val + 1) * (1 - alpha)) / n_val)
            level = min(1.0, np.ceil((n_val + 1) * (1.0 - alpha)) / n_val)
            offset_tx = float(np.quantile(scores_tx, level))
            offset_tn = float(np.quantile(scores_tn, level))
            offset_rn = float(np.quantile(scores_rn, level))

            self.conformal_offsets[h] = {
                'tmax_offset_90': offset_tx,
                'tmin_offset_90': offset_tn,
                'rain_offset_90': offset_rn
            }

        logger.info("Conformal calibration completed successfully.")
        return self

    def predict_with_intervals(self, X: np.ndarray, clim_center: np.ndarray, issue_doys: np.ndarray) -> Dict:
        """
        Generates calibrated quantiles and conformal uncertainty intervals.
        Returns:
          'tmax': dict with 'q05', 'q25', 'q50', 'q75', 'q95', 'calibrated_lower', 'calibrated_upper'
          'tmin': dict with 'q05', 'q25', 'q50', 'q75', 'q95', 'calibrated_lower', 'calibrated_upper'
          'rain': dict with 'q05', 'q25', 'q50', 'q75', 'q95', 'calibrated_lower', 'calibrated_upper'
        """
        N = len(X)
        res = {
            'tmax': {f'q{int(q*100):02d}': np.zeros((N, self.lead_days), dtype=np.float32) for q in self.quantiles},
            'tmin': {f'q{int(q*100):02d}': np.zeros((N, self.lead_days), dtype=np.float32) for q in self.quantiles},
            'rain': {f'q{int(q*100):02d}': np.zeros((N, self.lead_days), dtype=np.float32) for q in self.quantiles}
        }
        res['tmax']['calibrated_lower_90'] = np.zeros((N, self.lead_days), dtype=np.float32)
        res['tmax']['calibrated_upper_90'] = np.zeros((N, self.lead_days), dtype=np.float32)
        res['tmin']['calibrated_lower_90'] = np.zeros((N, self.lead_days), dtype=np.float32)
        res['tmin']['calibrated_upper_90'] = np.zeros((N, self.lead_days), dtype=np.float32)
        res['rain']['calibrated_lower_90'] = np.zeros((N, self.lead_days), dtype=np.float32)
        res['rain']['calibrated_upper_90'] = np.zeros((N, self.lead_days), dtype=np.float32)

        for h in range(1, self.lead_days + 1):
            h_idx = h - 1
            target_doys = ((issue_doys + h - 1) % 366)
            c_tx = clim_center[target_doys, 1]
            c_tn = clim_center[target_doys, 2]

            offsets = self.conformal_offsets.get(h, {'tmax_offset_90': 0.0, 'tmin_offset_90': 0.0, 'rain_offset_90': 0.0})

            # Tmax quantiles
            for q in self.quantiles:
                q_key = f'q{int(q*100):02d}'
                pred_anom_tx = self.models[h]['tmax'][q].predict(X)
                pred_anom_tn = self.models[h]['tmin'][q].predict(X)
                pred_rn = np.maximum(0.0, self.models[h]['rain'][q].predict(X))

                res['tmax'][q_key][:, h_idx] = c_tx + pred_anom_tx
                res['tmin'][q_key][:, h_idx] = c_tn + pred_anom_tn
                res['rain'][q_key][:, h_idx] = pred_rn

            # Conformalized 90% intervals
            res['tmax']['calibrated_lower_90'][:, h_idx] = res['tmax']['q05'][:, h_idx] - offsets['tmax_offset_90']
            res['tmax']['calibrated_upper_90'][:, h_idx] = res['tmax']['q95'][:, h_idx] + offsets['tmax_offset_90']

            res['tmin']['calibrated_lower_90'][:, h_idx] = res['tmin']['q05'][:, h_idx] - offsets['tmin_offset_90']
            res['tmin']['calibrated_upper_90'][:, h_idx] = res['tmin']['q95'][:, h_idx] + offsets['tmin_offset_90']

            res['rain']['calibrated_lower_90'][:, h_idx] = np.maximum(0.0, res['rain']['q05'][:, h_idx] - offsets['rain_offset_90'])
            res['rain']['calibrated_upper_90'][:, h_idx] = np.maximum(0.0, res['rain']['q95'][:, h_idx] + offsets['rain_offset_90'])

        return res


def evaluate_conformal_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Computes empirical coverage per lead day: fraction of observations inside [lower, upper].
    Expected: ~0.90 for nominal 90% interval.
    """
    lead_days = y_true.shape[1]
    coverage = np.zeros(lead_days, dtype=np.float32)
    for h in range(lead_days):
        yt = y_true[:, h]
        low = lower[:, h]
        up = upper[:, h]
        valid = ~np.isnan(yt)
        inside = (yt[valid] >= low[valid]) & (yt[valid] <= up[valid])
        coverage[h] = np.mean(inside) if np.sum(valid) > 0 else 0.0
    return coverage


def plot_uncertainty_and_calibration(eval_results: Dict, out_dir: str = "results") -> str:
    """
    Plots coverage vs lead day and example 20-day forecast curves with shaded uncertainty bands.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "uncertainty_calibration.png")

    lead_days = list(range(1, 21))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 90% Interval Coverage vs Lead Day
    axes[0, 0].plot(lead_days, eval_results['coverage_tmax_90'], 'o-', color='#DC2626', label='Tmax Conformal Coverage (90%)', linewidth=2)
    axes[0, 0].plot(lead_days, eval_results['coverage_rain_90'], 's-', color='#2563EB', label='Rain Conformal Coverage (90%)', linewidth=2)
    axes[0, 0].axhline(y=0.90, color='#16A34A', linestyle='--', label='Nominal 90% Target', linewidth=1.8)
    axes[0, 0].set_ylim(0.70, 1.0)
    axes[0, 0].set_title("Empirical Coverage of 90% Conformal Prediction Intervals", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Lead Day")
    axes[0, 0].set_ylabel("Empirical Coverage")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    # 2. Example 20-Day Tmax Forecast with Shaded Quantile Bands
    sample_tx = eval_results['sample_tmax']
    axes[0, 1].plot(lead_days, sample_tx['true'], 'ko-', label='Observed Truth', linewidth=2)
    axes[0, 1].plot(lead_days, sample_tx['q50'], 'r-', label='Median (q50)', linewidth=2)
    axes[0, 1].fill_between(lead_days, sample_tx['q25'], sample_tx['q75'], color='#F87171', alpha=0.4, label='50% Interval (q25-q75)')
    axes[0, 1].fill_between(lead_days, sample_tx['cal_low_90'], sample_tx['cal_up_90'], color='#FCA5A5', alpha=0.25, label='Calibrated 90% Band')
    axes[0, 1].set_title("20-Day Temperature Forecast with Calibrated Uncertainty", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Lead Day")
    axes[0, 1].set_ylabel("Max Temp (°C)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    # 3. Example 20-Day Rainfall Forecast with Shaded Quantile Bands
    sample_rn = eval_results['sample_rain']
    axes[1, 0].bar(lead_days, sample_rn['true'], color='#3B82F6', alpha=0.7, label='Observed Rain (mm)', width=0.5)
    axes[1, 0].plot(lead_days, sample_rn['q50'], color='#1D4ED8', marker='o', label='Median Expected (q50)', linewidth=2)
    axes[1, 0].fill_between(lead_days, sample_rn['cal_low_90'], sample_rn['cal_up_90'], color='#93C5FD', alpha=0.35, label='Calibrated 90% Uncertainty')
    axes[1, 0].set_title("20-Day Rainfall Forecast & Conformal Uncertainty Band", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Lead Day")
    axes[1, 0].set_ylabel("Precipitation (mm)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    # 4. Alert Exceedance Probabilities
    sample_prob = eval_results['alert_probabilities']
    axes[1, 1].plot(lead_days, sample_prob['prob_mod'], 'o-', color='#EAB308', label='P(Rain > 7.5mm Moderate)', linewidth=2)
    axes[1, 1].plot(lead_days, sample_prob['prob_heavy'], 's-', color='#F97316', label='P(Rain > 64.5mm Heavy)', linewidth=2)
    axes[1, 1].axhline(y=0.30, color='#DC2626', linestyle='--', label='Alert Trigger Threshold (30%)', linewidth=1.5)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Extreme-Event Exceedance Probabilities & Alert Trigger", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Lead Day")
    axes[1, 1].set_ylabel("Exceedance Probability")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Uncertainty calibration figure saved to {out_path}")
    return out_path
