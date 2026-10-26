"""
alerts.py - IMD Meteorological Extreme-Event Alerting System
Features:
- Official IMD Rainfall Classification Categories:
    - Light: < 7.5 mm
    - Moderate: 7.6 to 64.4 mm
    - Heavy: 64.5 to 115.5 mm
    - Very Heavy: > 115.6 mm
- Probabilistic exceedance calculation:
    Computes P(Rain >= threshold) using empirical CDF constructed from quantiles [q05, q25, q50, q75, q95]
- Multi-tier alert levels:
    - GREEN  (No warning / safe): P(moderate) < 0.30
    - YELLOW (Be Aware): P(moderate) >= 0.30 or P(heavy) in [0.20, 0.40)
    - ORANGE (Be Prepared): P(heavy) in [0.40, 0.70) or P(very_heavy) in [0.20, 0.50)
    - RED    (Take Action): P(heavy) >= 0.70 or P(very_heavy) >= 0.50
- Alert verification metrics: Hit Rate (POD), False Alarm Ratio (FAR), Critical Success Index (CSI), ETS
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import interp1d

# IMD official thresholds (mm)
IMD_THRESHOLDS = {
    'light_max': 7.5,
    'moderate_min': 7.6,
    'moderate_max': 64.4,
    'heavy_min': 64.5,
    'very_heavy_min': 115.6
}

ALERT_COLORS = {
    'GREEN': '#10B981',   # Emerald Green
    'YELLOW': '#F59E0B',  # Amber Yellow
    'ORANGE': '#F97316',  # Deep Orange
    'RED': '#EF4444'      # Crimson Red
}


def compute_exceedance_probability_from_quantiles(quantiles_dict: Dict[str, np.ndarray],
                                                 threshold: float) -> np.ndarray:
    """
    Computes P(Rain >= threshold) for each sample and lead day from predicted quantiles.
    quantiles_dict contains arrays for keys: 'q05', 'q25', 'q50', 'q75', 'q95'.
    Uses linear interpolation across quantile values to estimate the survival function: 1 - CDF(threshold).
    """
    q_levels = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    N, lead_days = quantiles_dict['q50'].shape
    exceed_prob = np.zeros((N, lead_days), dtype=np.float32)

    for i in range(N):
        for h in range(lead_days):
            q_vals = np.array([
                quantiles_dict['q05'][i, h],
                quantiles_dict['q25'][i, h],
                quantiles_dict['q50'][i, h],
                quantiles_dict['q75'][i, h],
                quantiles_dict['q95'][i, h]
            ])
            # Ensure monotonicity
            q_vals = np.maximum.accumulate(q_vals)

            if threshold <= q_vals[0]:
                # Exceeds lowest quantile
                prob = 0.98
            elif threshold >= q_vals[-1]:
                # Higher than 95th percentile: decay exponentially
                excess = threshold - q_vals[-1]
                prob = 0.05 * np.exp(-excess / max(10.0, q_vals[-1] * 0.5))
            else:
                # Interpolate CDF
                cdf_val = np.interp(threshold, q_vals, q_levels)
                prob = 1.0 - cdf_val

            exceed_prob[i, h] = float(np.clip(prob, 0.0, 1.0))

    return exceed_prob


def determine_imd_alert_level(p_moderate: float, p_heavy: float, p_very_heavy: float,
                              p_trigger: float = 0.30) -> Tuple[str, str, str]:
    """
    Determines alert badge, action recommendation, and CSS color given exceedance probabilities.
    Returns: (level, description, color)
    """
    if p_heavy >= 0.65 or p_very_heavy >= 0.40:
        return (
            "RED",
            "Severe weather warning: High probability of destructive rainfall (>64.5mm). Immediate protective action required.",
            ALERT_COLORS['RED']
        )
    elif p_heavy >= 0.35 or p_very_heavy >= 0.20:
        return (
            "ORANGE",
            "Moderate to heavy rain alert: Be prepared for waterlogging and disruption.",
            ALERT_COLORS['ORANGE']
        )
    elif p_moderate >= p_trigger or p_heavy >= 0.15:
        return (
            "YELLOW",
            "Weather advisory: Be aware of localized showers and wet conditions.",
            ALERT_COLORS['YELLOW']
        )
    else:
        return (
            "GREEN",
            "No active weather warnings. Conditions nominal.",
            ALERT_COLORS['GREEN']
        )


def evaluate_alert_contingency(y_true_binary: np.ndarray, alert_triggered_binary: np.ndarray) -> Dict[str, float]:
    """
    Computes alert verification metrics:
    - POD (Probability of Detection / Hit Rate) = Hits / (Hits + Misses)
    - FAR (False Alarm Ratio) = False Alarms / (Hits + False Alarms)
    - CSI (Critical Success Index / Threat Score) = Hits / (Hits + False Alarms + Misses)
    - ETS (Equitable Threat Score)
    """
    valid = (~np.isnan(y_true_binary)) & (~np.isnan(alert_triggered_binary))
    yt = y_true_binary[valid].astype(bool)
    yp = alert_triggered_binary[valid].astype(bool)

    hits = int(np.sum(yt & yp))
    false_alarms = int(np.sum((~yt) & yp))
    misses = int(np.sum(yt & (~yp)))
    correct_negatives = int(np.sum((~yt) & (~yp)))
    total = hits + false_alarms + misses + correct_negatives

    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    csi = float(hits / (hits + false_alarms + misses)) if (hits + false_alarms + misses) > 0 else 0.0

    # ETS
    hits_rand = ((hits + false_alarms) * (hits + misses)) / total if total > 0 else 0.0
    denom = hits + false_alarms + misses - hits_rand
    ets = float((hits - hits_rand) / denom) if denom > 0 else 0.0

    return {
        'hits': hits,
        'false_alarms': false_alarms,
        'misses': misses,
        'correct_negatives': correct_negatives,
        'POD_HitRate': round(pod, 4),
        'FAR_FalseAlarmRatio': round(far, 4),
        'CSI_ThreatScore': round(csi, 4),
        'ETS': round(ets, 4)
    }
