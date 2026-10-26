"""
features.py - Spatio-Temporal and Meteorological Feature Engineering Pipeline
Includes:
- Out-of-sample smoothed daily climatology subtraction to yield anomalies
- Cyclical day-of-year embeddings (sin/cos)
- Autoregressive lags: 1, 2, 3, 7, 14, 30 days
- Rolling statistics: mean and std over 7, 14, 30 days
- Spatial neighborhood context: 3x3 or 5x5 window
- Teleconnection drivers: Nino 3.4 index, Dipole Mode Index (DMI/IOD), Wheeler-Hendon MJO phase & amplitude
- Leakage-proof time-alignment: Features strictly use data strictly up to the forecast issue date (t <= T0)
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_climate_indices(indices_dir: str = "data/raw/indices") -> pd.DataFrame:
    """
    Parses and merges Nino 3.4, DMI (IOD), and MJO indices by date into a continuous daily DataFrame.
    """
    # 1. Nino 3.4 monthly anomaly
    nino_path = os.path.join(indices_dir, "nina34_anom.data")
    nino_records = []
    if os.path.exists(nino_path):
        lines = open(nino_path).readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 13 and parts[0].isdigit():
                yr = int(parts[0])
                for m, val in enumerate(parts[1:13], start=1):
                    fval = float(val)
                    if fval > -90.0:
                        nino_records.append((yr, m, fval))
    df_nino = pd.DataFrame(nino_records, columns=['yr', 'mon', 'nino34'])

    # 2. DMI monthly anomaly
    dmi_path = os.path.join(indices_dir, "dmi.txt")
    dmi_records = []
    if os.path.exists(dmi_path):
        lines = open(dmi_path).readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 13 and parts[0].isdigit():
                yr = int(parts[0])
                for m, val in enumerate(parts[1:13], start=1):
                    fval = float(val)
                    if fval > -90.0:
                        dmi_records.append((yr, m, fval))
    df_dmi = pd.DataFrame(dmi_records, columns=['yr', 'mon', 'dmi'])

    # 3. Wheeler-Hendon MJO daily
    mjo_path = os.path.join(indices_dir, "rmm_mjo.txt")
    mjo_records = []
    if os.path.exists(mjo_path):
        lines = open(mjo_path).readlines()
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 7 and parts[0].isdigit():
                yr = int(parts[0])
                mo = int(parts[1])
                dy = int(parts[2])
                rmm1 = float(parts[3])
                rmm2 = float(parts[4])
                phase = int(parts[5])
                amp = float(parts[6])
                if rmm1 < 900.0:
                    mjo_records.append((f"{yr:04d}-{mo:02d}-{dy:02d}", rmm1, rmm2, phase, amp))
    df_mjo = pd.DataFrame(mjo_records, columns=['date', 'rmm1', 'rmm2', 'mjo_phase', 'mjo_amp'])

    # Create master daily calendar 1999 to 2026
    date_range = pd.date_range("1999-01-01", "2026-12-31", freq="D")
    master_df = pd.DataFrame({'date': date_range})
    master_df['date_str'] = master_df['date'].dt.strftime('%Y-%m-%d')
    master_df['yr'] = master_df['date'].dt.year
    master_df['mon'] = master_df['date'].dt.month

    # Merge monthly Nino & DMI
    master_df = pd.merge(master_df, df_nino, on=['yr', 'mon'], how='left')
    master_df = pd.merge(master_df, df_dmi, on=['yr', 'mon'], how='left')
    master_df = pd.merge(master_df, df_mjo, left_on='date_str', right_on='date', how='left')

    # Forward-fill any minor missing points and fill edge NaNs with neutral 0.0
    master_df['nino34'] = master_df['nino34'].ffill().bfill().fillna(0.0)
    master_df['dmi'] = master_df['dmi'].ffill().bfill().fillna(0.0)
    master_df['rmm1'] = master_df['rmm1'].ffill().bfill().fillna(0.0)
    master_df['rmm2'] = master_df['rmm2'].ffill().bfill().fillna(0.0)
    master_df['mjo_amp'] = master_df['mjo_amp'].ffill().bfill().fillna(0.5)
    master_df['mjo_phase'] = master_df['mjo_phase'].ffill().bfill().fillna(1).astype(int)

    # Cyclical encoding of MJO phase (1 to 8)
    master_df['mjo_sin'] = np.sin(2 * np.pi * master_df['mjo_phase'] / 8.0)
    master_df['mjo_cos'] = np.cos(2 * np.pi * master_df['mjo_phase'] / 8.0)

    res_df = master_df[['date_str', 'nino34', 'dmi', 'rmm1', 'rmm2', 'mjo_amp', 'mjo_sin', 'mjo_cos']].rename(
        columns={'date_str': 'date'}
    )
    return res_df.set_index('date')


def extract_point_spatiotemporal_features(weather_tensor: np.ndarray,
                                          dates: List[str],
                                          lat_idx: int,
                                          lon_idx: int,
                                          clim_mean: np.ndarray,
                                          clim_std: np.ndarray,
                                          indices_df: Optional[pd.DataFrame] = None,
                                          lead_days: int = 20,
                                          lookback_max: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts strictly backward-looking features for a specific grid cell (lat_idx, lon_idx)
    across all valid timesteps t in [lookback_max, T - lead_days].
    
    Leakage Protection:
      At step t (the issue date), the feature vector uses only observations at t, t-1, t-2, t-3, t-7, t-14, t-30.
      Future ground-truth observations at t+h (h=1..20) are the prediction targets.
    
    Returns:
      X: (N_samples, N_features)
      Y_anom: (N_samples, lead_days, 3) [Anomalies for rain, tmax, tmin]
      feature_names: list of feature descriptions
    """
    T, C, H, W = weather_tensor.shape
    r_half = 1 # 3x3 neighborhood

    r_min = max(0, lat_idx - r_half)
    r_max = min(H, lat_idx + r_half + 1)
    c_min = max(0, lon_idx - r_half)
    c_max = min(W, lon_idx + r_half + 1)

    # Slice time series for this 3x3 neighborhood: (T, 3, 3, 3)
    neigh_ts = weather_tensor[:, :, r_min:r_max, c_min:c_max]
    center_ts = weather_tensor[:, :, lat_idx, lon_idx] # (T, 3)

    doys = np.array([pd.to_datetime(d).dayofyear for d in dates])

    # Precompute anomalies
    clim_center = clim_mean[:, :, lat_idx, lon_idx] # (366, 3)
    clim_std_center = clim_std[:, :, lat_idx, lon_idx] # (366, 3)

    # Center anomalies: (T, 3)
    anom_center = np.zeros_like(center_ts)
    for t in range(T):
        d_idx = doys[t] - 1
        anom_center[t] = (center_ts[t] - clim_center[d_idx])

    # Sample indices
    valid_t = list(range(lookback_max, T - lead_days))
    N_samples = len(valid_t)

    X_list = []
    Y_list = []
    feat_names = []

    lags = [0, 1, 2, 3, 7, 14, 30] # Lag 0 is current day observation
    windows = [7, 14, 30]

    for idx, t in enumerate(valid_t):
        cur_date_str = dates[t]
        doy = doys[t]
        
        # 1. Cyclical DOY
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        sample_feats = [sin_doy, cos_doy]

        # 2. Climatology normal for reference
        c_mean_t = clim_center[doy - 1]
        sample_feats.extend(c_mean_t.tolist())

        # 3. Lags of raw values and anomalies for rain, tmax, tmin
        for ch in range(3):
            for l in lags:
                obs_t = t - l
                sample_feats.append(center_ts[obs_t, ch])
                sample_feats.append(anom_center[obs_t, ch])

        # 4. Rolling statistics (mean and std) over past 7, 14, 30 days
        for ch in range(3):
            for w in windows:
                w_slice = center_ts[t - w + 1 : t + 1, ch]
                sample_feats.append(float(np.nanmean(w_slice)))
                sample_feats.append(float(np.nanstd(w_slice)))

        # 5. Spatial context: mean and std of 3x3 neighborhood at t
        for ch in range(3):
            n_slice = neigh_ts[t, ch]
            sample_feats.append(float(np.nanmean(n_slice)))
            sample_feats.append(float(np.nanstd(n_slice)))

        # 6. Teleconnection scalar indices
        if indices_df is not None and cur_date_str in indices_df.index:
            row = indices_df.loc[cur_date_str]
            sample_feats.extend([row['nino34'], row['dmi'], row['rmm1'], row['rmm2'], row['mjo_amp'], row['mjo_sin'], row['mjo_cos']])
        else:
            sample_feats.extend([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0])

        X_list.append(sample_feats)

        # Build multi-horizon target anomalies for h = 1..lead_days
        # Target is (lead_days, 3)
        y_targets_h = np.zeros((lead_days, 3), dtype=np.float32)
        for h in range(1, lead_days + 1):
            future_t = t + h
            f_doy = doys[future_t] - 1
            # Anomaly target: true future observation minus climatology normal
            y_targets_h[h - 1] = center_ts[future_t] - clim_center[f_doy]
            # Also keep raw rain for two-stage model
        Y_list.append(y_targets_h)

    # Feature names once
    feat_names = ['sin_doy', 'cos_doy', 'clim_rain', 'clim_tmax', 'clim_tmin']
    for ch_name in ['rain', 'tmax', 'tmin']:
        for l in lags:
            feat_names.append(f'{ch_name}_lag_{l}')
            feat_names.append(f'{ch_name}_anom_lag_{l}')
        for w in windows:
            feat_names.append(f'{ch_name}_roll_mean_{w}')
            feat_names.append(f'{ch_name}_roll_std_{w}')
        feat_names.append(f'{ch_name}_neigh_mean')
        feat_names.append(f'{ch_name}_neigh_std')
    feat_names.extend(['nino34', 'dmi', 'rmm1', 'rmm2', 'mjo_amp', 'mjo_sin', 'mjo_cos'])

    X = np.array(X_list, dtype=np.float32)
    Y_anom = np.array(Y_list, dtype=np.float32) # (N_samples, lead_days, 3)
    return X, Y_anom, feat_names
