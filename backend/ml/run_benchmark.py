"""
Run benchmark evaluation on v1.0 model across 20 lead days.
"""
import sys
import os
import pickle
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from backend.ml.evaluate import evaluate_forecasts_per_lead, compute_climatology, plot_skill_vs_lead_day

def main():
    print('Loading v1.0 data and model...')
    w22 = np.load('data/processed/weather_2022_1deg.npy')
    w23 = np.load('data/processed/weather_2023_1deg.npy')

    model_path = 'IMD_Weather_Predictor/model.pkl'
    if not os.path.exists(model_path):
        model_path = 'results/model.pkl'
    with open(model_path, 'rb') as f:
        model_v1 = pickle.load(f)

    dates_2022 = [d.strftime('%Y-%m-%d') for d in pd.date_range('2022-01-01', periods=365)]
    clim_mean, _ = compute_climatology(w22, dates_2022)

    T, C, H, W = w23.shape
    num_leads = 20
    test_t_indices = list(range(3, T - num_leads, 4))
    N_cases = len(test_t_indices)
    print(f'Running recursive forecast on {N_cases} test cases across 2023...')

    y_true_all = np.zeros((N_cases, num_leads, C, H, W), dtype=np.float32)
    y_pred_v1 = np.zeros((N_cases, num_leads, C, H, W), dtype=np.float32)
    y_clim_all = np.zeros((N_cases, num_leads, C, H, W), dtype=np.float32)
    y_persist_all = np.zeros((N_cases, num_leads, C, H, W), dtype=np.float32)

    lon_grid, lat_grid = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))

    for case_idx, start_t in enumerate(test_t_indices):
        t_base = start_t - 1
        persist_val = w23[t_base]
        
        y_true_all[case_idx] = w23[start_t : start_t + num_leads]
        
        for h in range(num_leads):
            cur_t = start_t + h
            doy = cur_t % 365
            y_clim_all[case_idx, h] = clim_mean[doy]
            y_persist_all[case_idx, h] = persist_val

        mem = []
        for l in range(3, 0, -1):
            pt = start_t - l
            r = np.maximum(0, np.nan_to_num(w23[pt, 0], nan=0.0))
            tx = np.nan_to_num(w23[pt, 1], nan=25.0)
            tn = np.nan_to_num(w23[pt, 2], nan=15.0)
            dtr = tx - tn
            lr = np.log1p(r)
            mem.append(np.stack([r, tx, tn, dtr, lr], axis=0))

        for h in range(num_leads):
            cur_t = start_t + h
            doy = cur_t % 365
            sin_doy = np.sin(2 * np.pi * doy / 365.25)
            cos_doy = np.cos(2 * np.pi * doy / 365.25)
            
            mem_arr = np.stack(mem[-3:], axis=0)
            padded = np.pad(mem_arr, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
            
            past_slices = []
            for l in range(3, 0, -1):
                past_idx = 3 - l
                for dy in range(3):
                    for dx in range(3):
                        neigh = padded[past_idx, :, dy:dy+H, dx:dx+W]
                        past_slices.append(neigh)
            spatial_feats = np.stack(past_slices, axis=0).reshape(-1, H, W).transpose(1, 2, 0).reshape(H*W, -1)
            loc_feats = np.stack([lat_grid.ravel(), lon_grid.ravel(), np.full(H*W, sin_doy), np.full(H*W, cos_doy)], axis=1)
            x_step = np.hstack([spatial_feats, loc_feats])
            
            pred_trans = model_v1.predict(x_step)
            p_lr = pred_trans[:, 0].reshape(H, W)
            p_tx = pred_trans[:, 1].reshape(H, W)
            p_tn = pred_trans[:, 2].reshape(H, W)
            p_r = np.maximum(0, np.expm1(p_lr))
            p_dtr = p_tx - p_tn
            
            p_tx = np.clip(p_tx, -15.0, 60.0)
            p_tn = np.clip(p_tn, -25.0, 50.0)
            
            y_pred_v1[case_idx, h, 0] = p_r
            y_pred_v1[case_idx, h, 1] = p_tx
            y_pred_v1[case_idx, h, 2] = p_tn
            
            mem.append(np.stack([p_r, p_tx, p_tn, p_dtr, p_lr], axis=0))

    print('Forecast generation complete. Computing verification metrics...')
    results = evaluate_forecasts_per_lead(y_true_all, y_pred_v1, y_clim_all, y_persist_all, lead_days=20)
    plot_path = plot_skill_vs_lead_day(results, out_dir='results')

    os.makedirs('results', exist_ok=True)
    with open('results/v1_benchmark_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved results/v1_benchmark_metrics.json and results/skill_vs_lead_day.png')
    c_day = results['summary']['climatology_crossover_day']
    print(f'Crossover Day where model stops beating Climatology: Day {c_day}')

    tmax_m = results["model"]["tmax"]
    tmax_p = results["persistence"]["tmax"]
    tmax_c = results["climatology"]["tmax"]
    rain_m = results["model"]["rain"]

    for d in [0, 2, 6, 9, 13, 19]:
        lead = d + 1
        print(f"\n--- Lead Day {lead} ---")
        print(f"  Model       : Tmax RMSE={tmax_m['rmse'][d]:.2f} C, ACC={tmax_m['acc'][d]:.3f}, Rain MAE={rain_m['mae'][d]:.2f} mm")
        print(f"  Persistence : Tmax RMSE={tmax_p['rmse'][d]:.2f} C, ACC={tmax_p['acc'][d]:.3f}")
        print(f"  Climatology : Tmax RMSE={tmax_c['rmse'][d]:.2f} C, ACC={tmax_c['acc'][d]:.3f}")

if __name__ == '__main__':
    main()
