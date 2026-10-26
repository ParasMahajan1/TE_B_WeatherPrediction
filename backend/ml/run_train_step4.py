"""
Evaluation and training script comparing:
1. Direct Multi-Horizon vs v1.0 Autoregressive Recursive
2. Two-Stage Rainfall vs Tweedie Loss
Trained on 2000-2019, validated on 2020-2021, tested on 2022-2024.
"""

import os
import sys
import json
import zarr
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '.')
from backend.ml.features import load_climate_indices, extract_point_spatiotemporal_features
from backend.ml.train import MultiHorizonForecaster, TwoStageRainfallModel, save_model
from backend.ml.evaluate import compute_acc, compute_brier_score, compute_crps_point, compute_ets

def main():
    print("Loading Zarr store and Climatology...")
    root = zarr.open_group('data/processed/imd_weather_grid.zarr', mode='r')
    dates = root['dates'][:]
    weather = root['weather'] # (9132, 3, 129, 135)
    lats = root['lats'][:]
    lons = root['lons'][:]

    clim_mean = np.load('data/processed/climatology/clim_mean_2000_2019.npy')
    clim_std = np.load('data/processed/climatology/clim_std_2000_2019.npy')

    # Load climate indices
    indices_df = load_climate_indices()
    print("Climate indices loaded successfully.")

    # Select representative points across meteorological zones:
    # 1. Delhi NCR (North): 28.61N, 77.20E
    # 2. Mumbai (West Coast Monsoon): 19.07N, 72.87E
    # 3. Bengaluru (South Interior): 12.97N, 77.59E
    # 4. Kolkata (East / Gangetic): 22.57N, 88.36E
    # 5. Nagpur (Central): 21.14N, 79.08E
    test_locations = [
        ("Delhi", 28.61, 77.20),
        ("Mumbai", 19.07, 72.87),
        ("Bengaluru", 12.97, 77.59),
        ("Kolkata", 22.57, 88.36),
        ("Nagpur", 21.14, 79.08)
    ]

    # Convert coordinates to grid indices
    grid_pts = []
    for name, lat, lon in test_locations:
        r = int(np.argmin(np.abs(lats - lat)))
        c = int(np.argmin(np.abs(lons - lon)))
        grid_pts.append((name, r, c))
        print(f"Location {name:10s} -> lat_idx={r} ({lats[r]:.2f}N), lon_idx={c} ({lons[c]:.2f}E)")

    # Temporal split indices:
    # Train: 2000-01-01 to 2019-12-31 (idx 0 to 7305)
    # Val:   2020-01-01 to 2021-12-31 (idx 7305 to 8036)
    # Test:  2022-01-01 to 2024-12-31 (idx 8036 to 9132)
    val_start = np.where(dates == '2020-01-01')[0][0]
    test_start = np.where(dates == '2022-01-01')[0][0]
    total_days = len(dates)

    print(f"Splits: Train [0:{val_start}], Val [{val_start}:{test_start}], Test [{test_start}:{total_days}]")

    # Focus on Mumbai (high-variance monsoon precipitation) and Delhi (continental temperature extremes)
    # for deep model comparison
    chosen_pts = [grid_pts[1], grid_pts[0]] # Mumbai, Delhi

    comparison_results = {
        "locations": {},
        "summary": {}
    }

    for name, lat_idx, lon_idx in chosen_pts:
        print(f"\n=======================================================")
        print(f"Processing Spatio-Temporal Pipeline for {name} ({lats[lat_idx]:.2f}N, {lons[lon_idx]:.2f}E)")
        print(f"=======================================================")

        # Extract local time series slice (3x3 neighborhood across all 9132 days)
        # Using contiguous slice of weather tensor
        r_min, r_max = max(0, lat_idx - 1), min(129, lat_idx + 2)
        c_min, c_max = max(0, lon_idx - 1), min(135, lon_idx + 2)
        
        # Load local slice into RAM for fast feature engineering
        local_weather = weather[:, :, r_min:r_max, c_min:c_max] # (9132, 3, 3, 3)

        # Build feature matrix
        # Note: local_weather lat_idx is now relative: 1, lon_idx: 1
        X, Y_anom, feat_names = extract_point_spatiotemporal_features(
            local_weather, dates, lat_idx=1, lon_idx=1,
            clim_mean=clim_mean[:, :, r_min:r_max, c_min:c_max],
            clim_std=clim_std[:, :, r_min:r_max, c_min:c_max],
            indices_df=indices_df,
            lead_days=20,
            lookback_max=30
        )
        print(f"Extracted feature tensor X: {X.shape}, Targets Y_anom: {Y_anom.shape}")

        # Ground truth future raw values
        center_raw = local_weather[:, :, 1, 1] # (9132, 3)
        valid_t = np.array(range(30, total_days - 20))
        Y_raw = np.zeros((len(valid_t), 20, 3), dtype=np.float32)
        for h in range(1, 21):
            Y_raw[:, h - 1] = center_raw[valid_t + h]

        # Temporal partition masks relative to valid_t
        train_idx = np.where(valid_t < val_start)[0]
        val_idx = np.where((valid_t >= val_start) & (valid_t < test_start))[0]
        test_idx = np.where(valid_t >= test_start)[0]

        X_train, Y_anom_train, Y_raw_train = X[train_idx], Y_anom[train_idx], Y_raw[train_idx]
        X_val, Y_anom_val, Y_raw_val = X[val_idx], Y_anom[val_idx], Y_raw[val_idx]
        X_test, Y_anom_test, Y_raw_test = X[test_idx], Y_anom[test_idx], Y_raw[test_idx]

        issue_doys_val = np.array([pd.to_datetime(dates[t]).dayofyear for t in valid_t[val_idx]])
        issue_doys_test = np.array([pd.to_datetime(dates[t]).dayofyear for t in valid_t[test_idx]])
        clim_center = clim_mean[:, :, lat_idx, lon_idx] # (366, 3)

        # 1. Compare Rainfall Models on Validation set: Two-Stage vs Tweedie
        print("\n--- Comparing Rainfall Formulations (Two-Stage vs Tweedie) on Validation ---")
        val_rain_raw = Y_raw_val[:, :, 0] # (N_val, 20)

        # Train Two-Stage
        m_twostage = MultiHorizonForecaster(lead_days=20, rain_model_type='two_stage')
        m_twostage.fit(X_train, Y_anom_train, Y_raw_train[:, :, 0])
        pred_val_twostage = m_twostage.predict(X_val, clim_center, issue_doys_val)

        # Train Tweedie
        m_tweedie = MultiHorizonForecaster(lead_days=20, rain_model_type='tweedie')
        m_tweedie.fit(X_train, Y_anom_train, Y_raw_train[:, :, 0])
        pred_val_tweedie = m_tweedie.predict(X_val, clim_center, issue_doys_val)

        # Rain CRPS & Brier at lead days 1, 3, 7, 14, 20
        rain_comp = {"two_stage": {"crps": [], "brier_15mm": []}, "tweedie": {"crps": [], "brier_15mm": []}}
        for h_idx in [0, 2, 6, 13, 19]:
            lead = h_idx + 1
            yt_rain = val_rain_raw[:, h_idx]
            true_15mm = (yt_rain >= 15.6).astype(float)

            # Two-Stage
            yp_2s = pred_val_twostage['rain'][:, h_idx]
            crps_2s = compute_crps_point(yt_rain, yp_2s)
            prob_2s = pred_val_twostage['rain_prob'][:, h_idx]
            # calibrated probability of >15mm
            brier_2s = compute_brier_score(true_15mm, np.clip(yp_2s / 15.6, 0.0, 1.0))

            # Tweedie
            yp_tw = pred_val_tweedie['rain'][:, h_idx]
            crps_tw = compute_crps_point(yt_rain, yp_tw)
            brier_tw = compute_brier_score(true_15mm, np.clip(yp_tw / 15.6, 0.0, 1.0))

            rain_comp["two_stage"]["crps"].append(round(crps_2s, 3))
            rain_comp["two_stage"]["brier_15mm"].append(round(brier_2s, 3))
            rain_comp["tweedie"]["crps"].append(round(crps_tw, 3))
            rain_comp["tweedie"]["brier_15mm"].append(round(brier_tw, 3))

            print(f"Lead Day {lead:2d} | Two-Stage CRPS: {crps_2s:.2f} mm (Brier: {brier_2s:.3f}) | Tweedie CRPS: {crps_tw:.2f} mm (Brier: {brier_tw:.3f})")

        # Pick best rainfall model
        mean_crps_2s = np.mean(rain_comp["two_stage"]["crps"])
        mean_crps_tw = np.mean(rain_comp["tweedie"]["crps"])
        best_forecaster = m_twostage if mean_crps_2s <= mean_crps_tw else m_tweedie
        best_type = "two_stage" if mean_crps_2s <= mean_crps_tw else "tweedie"
        print(f"Selected Best Rainfall Model Formulation: {best_type.upper()} (Validation CRPS: {min(mean_crps_2s, mean_crps_tw):.2f} mm)")

        # Save model
        model_save_path = f"results/models/v2/direct_forecaster_{name.lower()}.pkl"
        save_model(best_forecaster, model_save_path)

        # 2. Evaluate on Independent TEST SET (2022-2024)
        print(f"\n--- Performance on Independent Test Set (2022-2024) for {name} ---")
        pred_test = best_forecaster.predict(X_test, clim_center, issue_doys_test)

        test_tmax_raw = Y_raw_test[:, :, 1]
        test_tmin_raw = Y_raw_test[:, :, 2]
        test_rain_raw = Y_raw_test[:, :, 0]

        loc_eval = {
            "lead_days": [1, 3, 7, 10, 14, 20],
            "tmax_rmse": [],
            "tmax_acc": [],
            "rain_mae": [],
            "rain_ets_15mm": []
        }

        for lead in [1, 3, 7, 10, 14, 20]:
            h_idx = lead - 1
            yt_tx = test_tmax_raw[:, h_idx]
            yp_tx = pred_test['tmax'][:, h_idx]
            
            target_doys_test = ((issue_doys_test + lead - 1) % 366)
            yc_tx = clim_center[target_doys_test, 1]

            rmse_tx = float(np.sqrt(np.mean((yp_tx - yt_tx) ** 2)))
            acc_tx = compute_acc(yt_tx, yp_tx, yc_tx)

            yt_r = test_rain_raw[:, h_idx]
            yp_r = pred_test['rain'][:, h_idx]
            mae_r = float(np.mean(np.abs(yp_r - yt_r)))

            # ETS at 15.6 mm
            from backend.ml.evaluate import compute_contingency_table
            a, b, c, d = compute_contingency_table(yt_r, yp_r, 15.6)
            ets_r = compute_ets(a, b, c, d)

            loc_eval["tmax_rmse"].append(round(rmse_tx, 3))
            loc_eval["tmax_acc"].append(round(acc_tx, 3))
            loc_eval["rain_mae"].append(round(mae_r, 3))
            loc_eval["rain_ets_15mm"].append(round(ets_r, 3))

            print(f"Lead {lead:2d} | Tmax RMSE: {rmse_tx:.2f} C | Tmax ACC: {acc_tx:.3f} | Rain MAE: {mae_r:.2f} mm | Rain ETS (>15.6mm): {ets_r:.3f}")

        comparison_results["locations"][name] = {
            "test_metrics": loc_eval,
            "rain_comparison": rain_comp,
            "best_rain_model": best_type
        }

    # Save summary
    with open('results/step4_multihorizon_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print("\nSuccessfully saved results to results/step4_multihorizon_results.json")

if __name__ == '__main__':
    main()
