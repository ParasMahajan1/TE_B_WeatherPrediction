"""
Execution script for Step 5 (Uncertainty Quantification & Conformal Calibration)
and Step 6 (IMD Extreme-Event Alert Tuning & Verification).
Validates on 2020-2021, evaluates on 2022-2024 test set.
"""

import os
import sys
import json
import zarr
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from backend.ml.features import load_climate_indices, extract_point_spatiotemporal_features
from backend.ml.uncertainty import (
    ConformalQuantileForecaster, evaluate_conformal_coverage, plot_uncertainty_and_calibration
)
from backend.ml.alerts import (
    compute_exceedance_probability_from_quantiles, determine_imd_alert_level,
    evaluate_alert_contingency, IMD_THRESHOLDS
)

def main():
    print("Loading Zarr store and Climatology...")
    root = zarr.open_group('data/processed/imd_weather_grid.zarr', mode='r')
    dates = root['dates'][:]
    weather = root['weather']
    lats = root['lats'][:]
    lons = root['lons'][:]

    clim_mean = np.load('data/processed/climatology/clim_mean_2000_2019.npy')
    clim_std = np.load('data/processed/climatology/clim_std_2000_2019.npy')

    indices_df = load_climate_indices()
    print("Climate indices loaded.")

    # Test on Mumbai (Monsoon extreme rain) and Delhi (Continental temperature)
    cities = [
        ("Mumbai", 19.07, 72.87),
        ("Delhi", 28.61, 77.20)
    ]

    val_start = np.where(dates == '2020-01-01')[0][0]
    test_start = np.where(dates == '2022-01-01')[0][0]
    total_days = len(dates)

    step5_6_results = {}

    for name, lat, lon in cities:
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))
        print(f"\n=======================================================")
        print(f"STEP 5 & 6: Uncertainty & Alerts for {name} ({lats[lat_idx]:.2f}N, {lons[lon_idx]:.2f}E)")
        print(f"=======================================================")

        r_min, r_max = max(0, lat_idx - 1), min(129, lat_idx + 2)
        c_min, c_max = max(0, lon_idx - 1), min(135, lon_idx + 2)
        local_weather = weather[:, :, r_min:r_max, c_min:c_max]

        X, Y_anom, _ = extract_point_spatiotemporal_features(
            local_weather, dates, lat_idx=1, lon_idx=1,
            clim_mean=clim_mean[:, :, r_min:r_max, c_min:c_max],
            clim_std=clim_std[:, :, r_min:r_max, c_min:c_max],
            indices_df=indices_df,
            lead_days=20,
            lookback_max=30
        )

        center_raw = local_weather[:, :, 1, 1]
        valid_t = np.array(range(30, total_days - 20))
        Y_raw = np.zeros((len(valid_t), 20, 3), dtype=np.float32)
        for h in range(1, 21):
            Y_raw[:, h - 1] = center_raw[valid_t + h]

        train_idx = np.where(valid_t < val_start)[0]
        val_idx = np.where((valid_t >= val_start) & (valid_t < test_start))[0]
        test_idx = np.where(valid_t >= test_start)[0]

        X_train, Y_anom_train, Y_raw_train = X[train_idx], Y_anom[train_idx], Y_raw[train_idx]
        X_val, Y_anom_val, Y_raw_val = X[val_idx], Y_anom[val_idx], Y_raw[val_idx]
        X_test, Y_anom_test, Y_raw_test = X[test_idx], Y_anom[test_idx], Y_raw[test_idx]

        issue_doys_val = np.array([pd.to_datetime(dates[t]).dayofyear for t in valid_t[val_idx]])
        issue_doys_test = np.array([pd.to_datetime(dates[t]).dayofyear for t in valid_t[test_idx]])
        clim_center = clim_mean[:, :, lat_idx, lon_idx]

        # ---------------------------------------------------------
        # STEP 5: Fit Quantile Regressors and Calibrate with Conformal Prediction
        # ---------------------------------------------------------
        conformal_forecaster = ConformalQuantileForecaster(lead_days=20)
        conformal_forecaster.fit(X_train, Y_anom_train, Y_raw_train[:, :, 0])
        conformal_forecaster.calibrate(X_val, Y_anom_val, Y_raw_val[:, :, 0], nominal_coverage=0.90)

        # Predict on Test Set
        pred_test = conformal_forecaster.predict_with_intervals(X_test, clim_center, issue_doys_test)

        # Evaluate Coverage on Test Set
        test_tx_true = Y_raw_test[:, :, 1]
        test_rn_true = Y_raw_test[:, :, 0]

        cov_tx_90 = evaluate_conformal_coverage(
            test_tx_true, pred_test['tmax']['calibrated_lower_90'], pred_test['tmax']['calibrated_upper_90']
        )
        cov_rn_90 = evaluate_conformal_coverage(
            test_rn_true, pred_test['rain']['calibrated_lower_90'], pred_test['rain']['calibrated_upper_90']
        )

        print("\n--- Empirical Coverage of 90% Conformal Interval on Unseen Test Set (2022-2024) ---")
        for h in [1, 3, 7, 10, 14, 20]:
            print(f"Lead {h:2d} | Tmax 90% Coverage: {cov_tx_90[h-1]*100:.1f}% | Rain 90% Coverage: {cov_rn_90[h-1]*100:.1f}%")

        # ---------------------------------------------------------
        # STEP 6: Extreme Event Alerts & Contingency Verification
        # ---------------------------------------------------------
        # Exceedance probabilities for moderate (7.6mm) and heavy (64.5mm)
        p_moderate = compute_exceedance_probability_from_quantiles(pred_test['rain'], 7.6)
        p_heavy = compute_exceedance_probability_from_quantiles(pred_test['rain'], 64.5)
        p_very_heavy = compute_exceedance_probability_from_quantiles(pred_test['rain'], 115.6)

        print("\n--- Extreme-Event Alert Performance (Threshold: 30% trigger) ---")
        alert_metrics_mod = {}
        alert_metrics_hvy = {}

        for lead in [1, 3, 7, 10, 14, 20]:
            h_idx = lead - 1
            yt_rain = test_rn_true[:, h_idx]

            # Moderate Rain Alert (>7.5 mm)
            yt_mod_bin = (yt_rain >= 7.6).astype(int)
            alert_mod_bin = (p_moderate[:, h_idx] >= 0.30).astype(int)
            m_mod = evaluate_alert_contingency(yt_mod_bin, alert_mod_bin)
            alert_metrics_mod[f"lead_{lead}"] = m_mod

            # Heavy Rain Alert (>64.5 mm)
            yt_hvy_bin = (yt_rain >= 64.5).astype(int)
            alert_hvy_bin = (p_heavy[:, h_idx] >= 0.25).astype(int)
            m_hvy = evaluate_alert_contingency(yt_hvy_bin, alert_hvy_bin)
            alert_metrics_hvy[f"lead_{lead}"] = m_hvy

            print(f"Lead {lead:2d} | Mod Alert (>7.5mm)  : Hit Rate={m_mod['POD_HitRate']:.2f}, FAR={m_mod['FAR_FalseAlarmRatio']:.2f}, ETS={m_mod['ETS']:.3f}")
            if np.sum(yt_hvy_bin) > 0:
                print(f"        | Heavy Alert (>64.5mm): Hit Rate={m_hvy['POD_HitRate']:.2f}, FAR={m_hvy['FAR_FalseAlarmRatio']:.2f}, ETS={m_hvy['ETS']:.3f}")

        # Save sample forecast for plot
        sample_idx = 180 # Peak monsoon sample
        plot_data = {
            'coverage_tmax_90': cov_tx_90.tolist(),
            'coverage_rain_90': cov_rn_90.tolist(),
            'sample_tmax': {
                'true': test_tx_true[sample_idx].tolist(),
                'q50': pred_test['tmax']['q50'][sample_idx].tolist(),
                'q25': pred_test['tmax']['q25'][sample_idx].tolist(),
                'q75': pred_test['tmax']['q75'][sample_idx].tolist(),
                'cal_low_90': pred_test['tmax']['calibrated_lower_90'][sample_idx].tolist(),
                'cal_up_90': pred_test['tmax']['calibrated_upper_90'][sample_idx].tolist()
            },
            'sample_rain': {
                'true': test_rn_true[sample_idx].tolist(),
                'q50': pred_test['rain']['q50'][sample_idx].tolist(),
                'cal_low_90': pred_test['rain']['calibrated_lower_90'][sample_idx].tolist(),
                'cal_up_90': pred_test['rain']['calibrated_upper_90'][sample_idx].tolist()
            },
            'alert_probabilities': {
                'prob_mod': p_moderate[sample_idx].tolist(),
                'prob_heavy': p_heavy[sample_idx].tolist()
            }
        }

        fig_path = plot_uncertainty_and_calibration(plot_data, out_dir="results")

        # Save model
        model_save_path = f"results/models/v2/conformal_forecaster_{name.lower()}.pkl"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump(conformal_forecaster, f)
        print(f"Saved calibrated conformal model to {model_save_path}")

        step5_6_results[name] = {
            'tmax_coverage_90': [round(float(c), 3) for c in cov_tx_90],
            'rain_coverage_90': [round(float(c), 3) for c in cov_rn_90],
            'alert_metrics_moderate': alert_metrics_mod,
            'alert_metrics_heavy': alert_metrics_hvy
        }

    # Save summary
    with open('results/step5_6_uncertainty_alerts.json', 'w') as f:
        json.dump(step5_6_results, f, indent=2)
    print("\nSaved Step 5 & 6 verification results to results/step5_6_uncertainty_alerts.json")

if __name__ == '__main__':
    main()
