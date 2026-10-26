"""
Unit test strictly validating temporal isolation and feature leakage prevention.
Ensures that at any forecast issue date T0, no feature uses data from t > T0.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.ml.features import extract_point_spatiotemporal_features

def test_feature_temporal_leakage():
    # Construct a synthetic time series where days 0..49 have value 10.0,
    # and days 50..99 are corrupted with an artificial future spike (e.g., 9999.0)
    T = 100
    dates = [f"2020-01-{(i%30)+1:02d}" for i in range(T)]
    synthetic_weather = np.full((T, 3, 5, 5), 10.0, dtype=np.float32)
    
    # Introduce deliberate future spike at t >= 50
    spike_time = 50
    synthetic_weather[spike_time:, :, :, :] = 9999.0

    clim_mean = np.full((366, 3, 5, 5), 10.0, dtype=np.float32)
    clim_std = np.full((366, 3, 5, 5), 1.0, dtype=np.float32)

    X, Y, feat_names = extract_point_spatiotemporal_features(
        synthetic_weather, dates, lat_idx=2, lon_idx=2,
        clim_mean=clim_mean, clim_std=clim_std, indices_df=None,
        lead_days=10, lookback_max=30
    )

    # For all issue dates t before spike_time (valid_t < spike_time),
    # NO feature in X should contain 9999.0
    valid_t = list(range(30, T - 10))
    for idx, t in enumerate(valid_t):
        if t < spike_time:
            # Check maximum feature value in X[idx]
            max_feat = np.max(X[idx])
            assert max_feat < 1000.0, f"LEAKAGE DETECTED: Feature vector at issue time t={t} contains future spike value {max_feat}!"
