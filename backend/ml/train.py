"""
train.py - Direct Multi-Horizon Training Engine for IMD Weather Prediction v2.0
Replaces recursive forecasting with direct per-lead-day models (Lead h in 1..20).
Features:
- Direct Gradient Boosting (LightGBM / HistGradientBoosting) for Tmax and Tmin anomaly prediction
- Two-Stage Model for zero-inflated Rainfall:
    Stage 1: Binary Classifier for P(rain > 0.1 mm)
    Stage 2: Continuous Regressor for rain amount given rain [log1p space]
- Comparison against single Tweedie / Poisson loss models
- Strict temporal split (Train: 2000-2019, Validation: 2020-2021, Test: 2022-2024)
- Automated metric evaluation on validation and test sets
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, mean_squared_error, mean_absolute_error

from backend.ml.features import load_climate_indices, extract_point_spatiotemporal_features
from backend.ml.evaluate import compute_acc, compute_brier_score, compute_crps_point, compute_ets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TwoStageRainfallModel:
    """
    Two-stage model for zero-inflated, heavy-tailed precipitation:
    Stage 1: P(Rain > 0.1 mm) classifier
    Stage 2: Amount | Rain regressor (log1p transform)
    Expected value = P(Rain) * expm1(predicted_log_amount)
    """
    def __init__(self, classifier_params=None, regressor_params=None):
        c_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'n_estimators': 80,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        if classifier_params:
            c_params.update(classifier_params)
        self.clf = lgb.LGBMClassifier(**c_params)

        r_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'n_estimators': 80,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        if regressor_params:
            r_params.update(regressor_params)
        self.reg = lgb.LGBMRegressor(**r_params)

    def fit(self, X: np.ndarray, y_raw_rain: np.ndarray):
        # Stage 1: Rain occurrence (>= 0.1 mm)
        rain_occ = (y_raw_rain >= 0.1).astype(int)
        self.clf.fit(X, rain_occ)

        # Stage 2: Amount given rain (log1p)
        mask_rain = (y_raw_rain >= 0.1)
        if np.sum(mask_rain) > 5:
            X_rain = X[mask_rain]
            y_log_rain = np.log1p(y_raw_rain[mask_rain])
            self.reg.fit(X_rain, y_log_rain)
        else:
            self.reg.fit(X, np.zeros(len(X)))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)
        pred_log = self.reg.predict(X)
        pred_amount = np.maximum(0.0, np.expm1(pred_log))
        return prob * pred_amount


class MultiHorizonForecaster:
    """
    Direct multi-horizon meteorological forecasting engine.
    Trains dedicated models for lead days 1 to lead_days.
    """
    def __init__(self, lead_days: int = 20, rain_model_type: str = 'two_stage'):
        self.lead_days = lead_days
        self.rain_model_type = rain_model_type # 'two_stage' or 'tweedie'
        self.models_tmax = {} # lead_day -> LGBMRegressor
        self.models_tmin = {} # lead_day -> LGBMRegressor
        self.models_rain = {} # lead_day -> TwoStageRainfallModel or LGBMRegressor

    def fit(self, X_train: np.ndarray, Y_anom_train: np.ndarray, Y_raw_rain_train: np.ndarray):
        """
        Trains models across all lead horizons h in 1..lead_days.
        Y_anom_train: (N, lead_days, 3) [anom_rain, anom_tmax, anom_tmin]
        Y_raw_rain_train: (N, lead_days)
        """
        logger.info(f"Training direct multi-horizon forecaster for {self.lead_days} lead days (Rain: {self.rain_model_type})...")
        
        for h in range(1, self.lead_days + 1):
            h_idx = h - 1
            y_tmax = np.nan_to_num(Y_anom_train[:, h_idx, 1], nan=0.0)
            y_tmin = np.nan_to_num(Y_anom_train[:, h_idx, 2], nan=0.0)
            y_rain_raw = np.maximum(0.0, np.nan_to_num(Y_raw_rain_train[:, h_idx], nan=0.0))

            # 1. Tmax anomaly regressor
            m_tx = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=70,
                learning_rate=0.06,
                num_leaves=31,
                random_state=42 + h,
                verbose=-1
            )
            m_tx.fit(X_train, y_tmax)
            self.models_tmax[h] = m_tx

            # 2. Tmin anomaly regressor
            m_tn = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=70,
                learning_rate=0.06,
                num_leaves=31,
                random_state=42 + h,
                verbose=-1
            )
            m_tn.fit(X_train, y_tmin)
            self.models_tmin[h] = m_tn

            # 3. Rainfall model
            if self.rain_model_type == 'two_stage':
                m_rain = TwoStageRainfallModel()
                m_rain.fit(X_train, y_rain_raw)
            else: # Tweedie
                if np.sum(y_rain_raw) <= 0.0:
                    y_rain_raw = y_rain_raw + 1e-4 # Tweedie requires strictly non-zero sum
                m_rain = lgb.LGBMRegressor(
                    objective='tweedie',
                    tweedie_variance_power=1.5,
                    n_estimators=70,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42 + h,
                    verbose=-1
                )
                m_rain.fit(X_train, y_rain_raw)
            self.models_rain[h] = m_rain

        logger.info("Direct multi-horizon training completed.")
        return self

    def predict(self, X: np.ndarray, clim_center: np.ndarray, issue_doys: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Inference: Predicts anomaly and adds climatology back.
        Returns dictionary containing point forecasts of shape (N, lead_days):
          'tmax': physical max temperature in degC
          'tmin': physical min temperature in degC
          'rain': physical rainfall in mm
          'rain_prob': precipitation probability
        """
        N = len(X)
        pred_tmax = np.zeros((N, self.lead_days), dtype=np.float32)
        pred_tmin = np.zeros((N, self.lead_days), dtype=np.float32)
        pred_rain = np.zeros((N, self.lead_days), dtype=np.float32)
        pred_rain_prob = np.zeros((N, self.lead_days), dtype=np.float32)

        for h in range(1, self.lead_days + 1):
            h_idx = h - 1
            # Climatology for future day
            target_doys = ((issue_doys + h - 1) % 366) # 0-indexed DOY
            c_tx = clim_center[target_doys, 1]
            c_tn = clim_center[target_doys, 2]

            # Anomaly predictions
            anom_tx = self.models_tmax[h].predict(X)
            anom_tn = self.models_tmin[h].predict(X)

            # Reconstructed values
            pred_tmax[:, h_idx] = c_tx + anom_tx
            pred_tmin[:, h_idx] = c_tn + anom_tn

            # Rainfall
            if self.rain_model_type == 'two_stage':
                r_model = self.models_rain[h]
                pred_rain[:, h_idx] = r_model.predict(X)
                pred_rain_prob[:, h_idx] = r_model.predict_proba(X)
            else:
                r_model = self.models_rain[h]
                p_r = r_model.predict(X)
                pred_rain[:, h_idx] = np.maximum(0.0, p_r)
                pred_rain_prob[:, h_idx] = np.clip(pred_rain[:, h_idx] / 5.0, 0.0, 1.0)

        # Enforce physical consistency (Tmax >= Tmin)
        pred_tmax = np.maximum(pred_tmax, pred_tmin + 0.5)
        pred_rain = np.maximum(0.0, pred_rain)

        return {
            'tmax': pred_tmax,
            'tmin': pred_tmin,
            'rain': pred_rain,
            'rain_prob': pred_rain_prob
        }


def save_model(forecaster: MultiHorizonForecaster, path: str = "results/models/v2/direct_forecaster.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(forecaster, f)
    logger.info(f"Saved direct multi-horizon model to {path}")


def load_model(path: str = "results/models/v2/direct_forecaster.pkl") -> MultiHorizonForecaster:
    with open(path, 'rb') as f:
        return pickle.load(f)
