# IMD Weather Prediction System v1.1

> **Operational High-Resolution Machine Learning Meteorological Forecaster**  
> Direct Multi-Horizon Forecasting (1–20 Days) | Conformal Uncertainty Bands | IMD Extreme Alert Matrix

---

## Key Improvements from v1.0 to v1.1

| Feature / Metric | v1.0 Legacy Baseline | v2.0 Operational System | Operational Impact |
|:---|:---|:---|:---|
| **Spatial Resolution** | 1.0° (~111 km, box-averaged) | **0.25° (~27 km native grid)** | 4x finer spatial resolution preserves microclimates & topography |
| **Data Span & Split** | 2022–2023 (2 years, arbitrary) | **2000–2024 (25 years, strict split)** | Train: 2000–2019, Val: 2020–2021, Test: 2022–2024 |
| **Forecast Strategy** | Recursive rollout ($t+1 \to t+2$) | **Direct Multi-Horizon per Lead Day** | **Zero error compounding** over 20-day horizons |
| **Target Formulation** | Raw absolute values | **Climatological Anomalies ($\Delta y$)** | Eliminates trivial seasonality drift; reconstructs at inference |
| **Precipitation Handling**| Single regressor (drizzles everywhere) | **Two-Stage Hurdle Classifier $\times$ Regressor** | Correctly models zero-inflation and extreme monsoon tails |
| **Uncertainty Quantification**| None (blind point predictions) | **Split Conformal CQR (90% & 50% intervals)** | **Empirical test coverage matches nominal 90%** |
| **Extreme Weather Alerts**| None | **Operational IMD Matrix (Green/Yellow/Orange/Red)** | Calibrated probability of exceedance ($>7.5$mm, $>64.5$mm, $>115.5$mm) |
| **Day 20 Tmax RMSE** | 3.46 °C (Worse than Climatology) | **1.33 °C (Statistically beats Climatology)** | **-61.6% reduction in long-lead forecast error** |

---

## Repository Architecture

```
TE_B_WeatherPrediction/
├── backend/
│   ├── app/
│   │   └── main.py              # Production FastAPI REST backend
│   ├── ml/
│   │   ├── data.py              # 25-year IMD download, bilinear regridding, Zarr builder
│   │   ├── features.py          # Lags, rolling windows, teleconnections (ENSO, IOD, MJO)
│   │   ├── train.py             # Direct LightGBM multi-horizon & Two-Stage rain model
│   │   ├── uncertainty.py       # Quantile regression & Conformal CQR calibration
│   │   ├── alerts.py            # IMD rainfall warning classifier & contingency verifier
│   │   ├── evaluate.py          # WMO evaluation metrics (RMSE, MAE, ACC, CRPS, ETS)
│   │   └── deep_model.py        # PyTorch ConvLSTM Spatio-Temporal Model & Gate Check
│   └── configs/
│       └── config.yaml          # Unified system configuration
├── frontend/
│   └── app.py                   # Streamlit + Folium + Plotly interactive dashboard
├── data/
│   ├── raw/                     # Original IMD binaries & teleconnection indices
│   └── processed/
│       ├── imd_weather_grid.zarr/ # Unified 25-year (9,132 days, 129x135) store
│       └── climatology/          # 2000-2019 strictly leak-free smoothed climatology
├── results/
│   ├── models/v2/               # Saved trained LightGBM & Conformal forecasters
│   ├── v1_benchmark_metrics.json
│   ├── step5_6_uncertainty_alerts.json
│   └── uncertainty_calibration.png
└── tests/                       # Automated unit & data leakage regression tests
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
# Core dependencies: lightgbm, zarr, xarray, torch, fastapi, uvicorn, streamlit, folium, streamlit-folium, plotly
```

### 2. Run Automated Regression & Leakage Tests
```bash
pytest tests/
```

### 3. Launch Backend API (FastAPI)
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation: `http://127.0.0.1:8000/docs`

### 4. Launch Interactive Frontend Dashboard (Streamlit)
```bash
streamlit run frontend/app.py
```

---

## Scientific Methodology & Gate Checks

1. **Strict Temporal Integrity**:
   - Training ($2000–2019$), Validation ($2020–2021$), and Testing ($2022–2024$) splits are strictly maintained without overlapping windows.
   - All climatological normals ($\mu_{\text{clim}}, \sigma_{\text{clim}}$) are derived exclusively from the 2000–2019 train set.
2. **Deep Learning Gate Check**:
   - A PyTorch ConvLSTM Spatio-Temporal model was evaluated against tabular LightGBM Direct Forecasters.
   - **Result**: LightGBM outperformed ConvLSTM on CPU validation (Day 1 Tmax RMSE: $1.28$°C vs $1.53$°C; Day 7: $1.76$°C vs $2.20$°C). Per the design gate rule, LightGBM + Conformal Prediction was selected as the production forecaster.
