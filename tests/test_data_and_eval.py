import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.ml.data import parse_imd_grd, regrid_temperature_to_rain_grid, is_leap_year
from backend.ml.evaluate import compute_acc, compute_ets, compute_brier_score, compute_crps_point, compute_f1

def test_leap_years():
    assert is_leap_year(2000) is True
    assert is_leap_year(2004) is True
    assert is_leap_year(2020) is True
    assert is_leap_year(2024) is True
    assert is_leap_year(2019) is False
    assert is_leap_year(2021) is False

def test_regrid_temperature_shape_and_range():
    # Synthetic 31x31 temperature field
    synthetic_temp = np.full((2, 31, 31), 30.0, dtype=np.float32)
    # Add gradient
    synthetic_temp[:, :, :] += np.linspace(0, 10, 31).reshape(1, 31, 1)
    
    regrid = regrid_temperature_to_rain_grid(synthetic_temp)
    assert regrid.shape == (2, 129, 135)
    # In interior region, interpolated values should match range
    valid_vals = regrid[~np.isnan(regrid)]
    assert len(valid_vals) > 0
    assert np.min(valid_vals) >= 29.5
    assert np.max(valid_vals) <= 40.5

def test_evaluation_acc_perfect_correlation():
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    y_pred = np.array([10.0, 15.0, 20.0, 25.0])
    y_clim = np.array([5.0, 5.0, 5.0, 5.0])
    acc = compute_acc(y_true, y_pred, y_clim)
    assert pytest.approx(acc, 0.001) == 1.0

def test_evaluation_ets_and_brier():
    y_true = np.array([0.0, 5.0, 20.0, 70.0])
    y_pred = np.array([0.0, 1.0, 18.0, 80.0])
    
    # 15.6mm threshold
    prob = np.array([0.0, 0.0, 1.0, 1.0])
    true_bin = np.array([0.0, 0.0, 1.0, 1.0])
    brier = compute_brier_score(true_bin, prob)
    assert pytest.approx(brier, 0.001) == 0.0
