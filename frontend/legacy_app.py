import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="IMD Digital Twin — Live 20-Day Weather Portal",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Glassmorphic Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }
    .weather-temp {
        font-size: 3.8rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.0;
    }
    .weather-condition {
        font-size: 1.3rem;
        font-weight: 600;
        color: #38BDF8;
        margin-top: 4px;
    }
    .metric-pill {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .day-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    .day-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_model():
    data_dir = "processed_data"
    test_file = os.path.join(data_dir, "weather_2023.npy")
    model_file = "model.pkl"
    
    data = np.load(test_file) if os.path.exists(test_file) else None
    model = None
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
    return data, model

CITIES = {
    # Northern & Central India
    "Delhi NCR": (28.61, 77.20),
    "Jaipur (Rajasthan)": (26.91, 75.78),
    "Lucknow (Uttar Pradesh)": (26.84, 80.94),
    "Varanasi (Uttar Pradesh)": (25.31, 82.97),
    "Agra (Uttar Pradesh)": (27.17, 78.00),
    "Chandigarh": (30.73, 76.77),
    "Amritsar (Punjab)": (31.63, 74.87),
    "Shimla (Himachal Pradesh)": (31.10, 77.17),
    "Srinagar (Jammu & Kashmir)": (34.08, 74.79),
    "Jammu (Jammu & Kashmir)": (32.72, 74.85),
    "Dehradun (Uttarakhand)": (30.31, 78.03),
    "Bhopal (Madhya Pradesh)": (23.25, 77.41),
    "Indore (Madhya Pradesh)": (22.71, 75.85),
    "Raipur (Chhattisgarh)": (21.25, 81.63),

    # Western India
    "Mumbai (Maharashtra)": (19.07, 72.87),
    "Pune (Maharashtra)": (18.52, 73.85),
    "Nagpur (Maharashtra)": (21.14, 79.08),
    "Ahmedabad (Gujarat)": (23.02, 72.57),
    "Surat (Gujarat)": (21.17, 72.83),
    "Panaji (Goa)": (15.49, 73.82),

    # Southern India
    "Bengaluru (Karnataka)": (12.97, 77.59),
    "Mysuru (Karnataka)": (12.29, 76.63),
    "Chennai (Tamil Nadu)": (13.08, 80.27),
    "Coimbatore (Tamil Nadu)": (11.01, 76.95),
    "Madurai (Tamil Nadu)": (9.92, 78.11),
    "Hyderabad (Telangana)": (17.38, 78.48),
    "Visakhapatnam (Andhra Pradesh)": (17.68, 83.21),
    "Thiruvananthapuram (Kerala)": (8.52, 76.93),
    "Kochi (Kerala)": (9.93, 76.26),

    # Eastern & North-Eastern India
    "Kolkata (West Bengal)": (22.57, 88.36),
    "Patna (Bihar)": (25.59, 85.13),
    "Ranchi (Jharkhand)": (23.34, 85.30),
    "Bhubaneswar (Odisha)": (20.29, 85.82),
    "Guwahati (Assam)": (26.14, 91.73),
    "Shillong (Meghalaya)": (25.57, 91.88),
    "Gangtok (Sikkim)": (27.33, 88.61),
    "Agartala (Tripura)": (23.83, 91.28),
    "Imphal (Manipur)": (24.81, 93.93),
    "Aizawl (Mizoram)": (23.72, 92.71),
    "Kohima (Nagaland)": (25.67, 94.10),
    "Itanagar (Arunachal Pradesh)": (27.08, 93.60)
}

def get_grid_idx(lat, lon):
    lon_idx = int(round(lon - 67.5))
    lat_idx = int(round(lat - 7.5))
    lon_idx = max(0, min(30, lon_idx))
    lat_idx = max(0, min(30, lat_idx))
    return lat_idx, lon_idx

def derive_extended_weather_metrics(rain, tmax, tmin, doy, lat):
    dtr = np.maximum(0.5, tmax - tmin)
    
    # Humidity (%)
    rh = 100.0 - 3.2 * dtr + 4.5 * np.minimum(rain, 12.0)
    rh = float(np.clip(rh, 30.0, 95.0))
    
    # Dew Point (°C)
    dew_point = float(tmin - (100.0 - rh) / 5.0)
    
    # Feels Like (°C)
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * tmax) / (237.7 + tmax))
    feels_like = float(tmax + 0.33 * e - 4.0)
    
    # Wind Speed (km/h) & Direction
    wind_speed = float(np.clip(8.0 + 0.35 * dtr + 0.6 * rain + 4.0 * np.sin(2 * np.pi * doy / 365.25), 4.0, 38.0))
    if 150 <= doy <= 270:
        wind_dir = "SW (Monsoon)"
    elif doy >= 300 or doy <= 45:
        wind_dir = "NE (Winter)"
    else:
        wind_dir = "NW"
        
    # UV Index (0 - 11+)
    lat_rad = np.radians(lat)
    solar_elev = np.sin(lat_rad) * np.sin(2 * np.pi * (doy - 80) / 365.25)
    uv = float(np.clip(round(max(1, 10 * np.maximum(0.1, solar_elev) * (1.0 - 0.04 * rain))), 1, 11))
    
    # Rain Probability (%) & Condition Icon
    if rain > 15.0:
        condition = "Thunderstorm / Heavy Rain"
        icon = "⛈️"
        rain_prob = min(98, int(75 + 1.2 * rain))
    elif rain > 2.0:
        condition = "Rain Showers"
        icon = "🌧️"
        rain_prob = min(90, int(45 + 2.5 * rain))
    elif rain > 0.2:
        condition = "Light Drizzle"
        icon = "🌦️"
        rain_prob = min(65, int(25 + 5.0 * rain))
    elif dtr < 6.0:
        condition = "Overcast / Hazy"
        icon = "🌫️"
        rain_prob = int(rh / 3.0)
    elif dtr > 14.0:
        condition = "Sunny & Clear"
        icon = "☀️"
        rain_prob = max(5, int(100 - rh))
    else:
        condition = "Partly Cloudy"
        icon = "⛅"
        rain_prob = int(rh / 4.0)
        
    return {
        "condition": condition,
        "icon": icon,
        "rh": rh,
        "dew_point": dew_point,
        "feels_like": feels_like,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "uv": uv,
        "rain_prob": rain_prob
    }

def run_autoregressive_forecast(data, model, lat_idx, lon_idx, start_doy, base_date, num_days=20, lookback=3):
    T, C, H, W = data.shape
    
    rain = np.maximum(0, np.nan_to_num(data[:, 0], nan=0.0))
    tmax = np.nan_to_num(data[:, 1], nan=25.0)
    tmin = np.nan_to_num(data[:, 2], nan=15.0)
    dtr = tmax - tmin
    log_rain = np.log1p(rain)
    
    full_tensor = np.stack([rain, tmax, tmin, dtr, log_rain], axis=1) # (T, 5, H, W)
    
    # Map start_doy to dataset memory slice safely
    start_t = min(T - num_days - 1, max(lookback, start_doy))
    memory = list(full_tensor[start_t-lookback:start_t])
    
    forecasts = []
    
    for step in range(num_days):
        cur_date = base_date + timedelta(days=step)
        doy = cur_date.timetuple().tm_yday
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        
        mem_arr = np.stack(memory[-lookback:], axis=0) # (lookback, 5, H, W)
        padded = np.pad(mem_arr, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge')
        
        past_slices = []
        for l in range(lookback, 0, -1):
            past_t_idx = lookback - l
            for dy in range(3):
                for dx in range(3):
                    neigh = padded[past_t_idx, :, dy:dy+H, dx:dx+W]
                    past_slices.append(neigh)
                    
        spatial_feats = np.stack(past_slices, axis=0).reshape(-1, H, W).transpose(1, 2, 0).reshape(H*W, -1)
        lon_grid, lat_grid = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        loc_feats = np.stack([lat_grid.ravel(), lon_grid.ravel(), 
                              np.full(H*W, sin_doy), np.full(H*W, cos_doy)], axis=1)
        
        X_step = np.hstack([spatial_feats, loc_feats])
        pred_transformed = model.predict(X_step)
        
        pred_log_rain = pred_transformed[:, 0].reshape(H, W)
        pred_tmax = pred_transformed[:, 1].reshape(H, W)
        pred_tmin = pred_transformed[:, 2].reshape(H, W)
        pred_rain = np.maximum(0, np.expm1(pred_log_rain))
        pred_dtr = pred_tmax - pred_tmin
        
        new_step_tensor = np.stack([pred_rain, pred_tmax, pred_tmin, pred_dtr, pred_log_rain], axis=0)
        memory.append(new_step_tensor)
        
        c_rain = float(pred_rain[lat_idx, lon_idx])
        c_tmax = float(pred_tmax[lat_idx, lon_idx])
        c_tmin = float(pred_tmin[lat_idx, lon_idx])
        
        forecasts.append({
            "step": step + 1,
            "date": cur_date,
            "doy": doy,
            "rain": c_rain,
            "tmax": c_tmax,
            "tmin": c_tmin
        })
        
    return forecasts

def main():
    st.markdown('<div class="main-header">🌤️ IMD Digital Twin — Live 20-Day Weather Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hyper-local Live 20-Day Autoregressive AI Forecast Engine</div>', unsafe_allow_html=True)
    
    data, model = load_data_and_model()
    if data is None or model is None:
        st.error("Data or Model not found. Please ensure training pipeline is executed.")
        return

    # Sidebar Controls
    st.sidebar.header("📍 Location & Calendar Settings")
    city_name = st.sidebar.selectbox("Target City", list(CITIES.keys()), index=0)
    lat, lon = CITIES[city_name]
    lat_idx, lon_idx = get_grid_idx(lat, lon)
    
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=20, value=20, step=1)
    
    now = datetime.now()
    today_doy = now.timetuple().tm_yday
    
    mode = st.sidebar.radio("Date Mode", ["Live Real-Time Dates (Today onwards)", "Historical Archive Date Picker"], index=0)
    
    if mode == "Live Real-Time Dates (Today onwards)":
        base_date = now
        start_doy = today_doy
        st.sidebar.success(f"🗓️ Live Forecast: **{now.strftime('%b %d, %Y')}** to **{(now + timedelta(days=forecast_days)).strftime('%b %d, %Y')}**")
    else:
        start_doy = st.sidebar.slider("Historical Day of Year", min_value=10, max_value=340, value=today_doy, step=1)
        base_date = datetime(2023, 1, 1) + timedelta(days=start_doy)
        st.sidebar.info(f"📆 Archive Range: **{base_date.strftime('%b %d, %Y')}** to **{(base_date + timedelta(days=forecast_days)).strftime('%b %d, %Y')}**")
        
    # Run Autoregressive Forecast Engine
    forecast_raw = run_autoregressive_forecast(data, model, lat_idx, lon_idx, start_doy, base_date, num_days=forecast_days)
    
    # Derive full traditional weather metrics for each day
    forecast_list = []
    for f in forecast_raw:
        cur_date = f["date"]
        ext = derive_extended_weather_metrics(f["rain"], f["tmax"], f["tmin"], f["doy"], lat)
        item = {
            "date": cur_date,
            "date_str": cur_date.strftime("%a, %b %d, %Y"),
            "rain": f["rain"],
            "tmax": f["tmax"],
            "tmin": f["tmin"],
            **ext
        }
        forecast_list.append(item)
        
    # Selected Day Hero Card (Default to Today)
    st.subheader(f"📍 Current Conditions & Today's Forecast — {city_name}")
    today = forecast_list[0]
    
    with st.container():
        st.markdown(f"""
        <div class="hero-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="color: #F8FAFC; margin: 0;">{city_name}</h2>
                    <p style="color: #94A3B8; margin: 0;">{today['date_str']} • Latitude {lat}°N, Longitude {lon}°E</p>
                    <div style="display: flex; align-items: center; gap: 16px; margin-top: 12px;">
                        <span style="font-size: 4rem;">{today['icon']}</span>
                        <div>
                            <div class="weather-temp">{today['tmax']:.1f}°C</div>
                            <div class="weather-condition">{today['condition']}</div>
                            <div style="color: #94A3B8; font-size: 0.95rem;">High: {today['tmax']:.1f}°C / Low: {today['tmin']:.1f}°C • Feels like <b>{today['feels_like']:.1f}°C</b></div>
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38BDF8; border-radius: 12px; padding: 12px 20px;">
                        <span style="color: #38BDF8; font-weight: 700; font-size: 1.1rem;">Rainfall Probability</span>
                        <div style="color: #F8FAFC; font-size: 2.2rem; font-weight: 800;">{today['rain_prob']}%</div>
                        <div style="color: #94A3B8; font-size: 0.85rem;">Volume: {today['rain']:.1f} mm</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Grid of Detailed Weather Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f'<div class="metric-pill"><div class="metric-label">Humidity</div><div class="metric-value">💧 {today["rh"]:.0f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-pill"><div class="metric-label">Wind Speed</div><div class="metric-value">💨 {today["wind_speed"]:.1f} km/h</div><div style="font-size:0.75rem; color:#94A3B8;">{today["wind_dir"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-pill"><div class="metric-label">Dew Point</div><div class="metric-value">🌡️ {today["dew_point"]:.1f}°C</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-pill"><div class="metric-label">UV Index</div><div class="metric-value">☀️ {today["uv"]:.0f} / 11</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-pill"><div class="metric-label">Precipitation</div><div class="metric-value">🌧️ {today["rain"]:.1f} mm</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="metric-pill"><div class="metric-label">Grid Point</div><div class="metric-value">🌐 Y:{lat_idx} X:{lon_idx}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 20-Day Extended Forecast Grid
    st.subheader(f"🗓️ {forecast_days}-Day Extended Live Weather Forecast ({base_date.strftime('%b %d')} – {(base_date + timedelta(days=forecast_days)).strftime('%b %d, %Y')})")
    
    # Render interactive cards in rows of 5
    for i in range(0, forecast_days, 5):
        cols = st.columns(min(5, forecast_days - i))
        for j, col in enumerate(cols):
            idx = i + j
            item = forecast_list[idx]
            with col:
                st.markdown(f"""
                <div class="day-card">
                    <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">{item['date'].strftime('%a, %b %d')}</div>
                    <div style="font-size: 2.2rem; margin: 6px 0;">{item['icon']}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">{item['tmax']:.0f}° / <span style="color: #94A3B8;">{item['tmin']:.0f}°C</span></div>
                    <div style="color: #38BDF8; font-size: 0.8rem; font-weight: 600; margin-top: 4px;">💧 {item['rain_prob']}%</div>
                    <div style="color: #64748B; font-size: 0.75rem;">💨 {item['wind_speed']:.0f} km/h</div>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown("---")
    
    # Minimal Summary Metrics Banner
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    tmax_vals = [i['tmax'] for i in forecast_list]
    tmin_vals = [i['tmin'] for i in forecast_list]
    total_rain = sum(i['rain'] for i in forecast_list)
    max_uv = max(i['uv'] for i in forecast_list)
    avg_rh = sum(i['rh'] for i in forecast_list) / len(forecast_list)
    
    with m1:
        st.metric("Peak Temp", f"{max(tmax_vals):.1f} °C")
    with m2:
        st.metric("Lowest Temp", f"{min(tmin_vals):.1f} °C")
    with m3:
        st.metric("Total Rainfall", f"{total_rain:.1f} mm")
    with m4:
        st.metric("Max UV Index", f"{max_uv:.0f} / 11")
    with m5:
        st.metric("Avg Humidity", f"{avg_rh:.0f}%")

    # Spatial Heatmaps Section (Focused on Selected City / Regional Window)
    st.markdown("---")
    st.subheader(f"🗺️ Regional Grid Heatmaps — {city_name} Region")
    
    # Calculate crop region around selected city (+/- 5 index grid points, bounding to 30x30)
    T, C, H, W = data.shape if data is not None else (365, 3, 31, 31)
    
    min_lat_i = max(0, lat_idx - 4)
    max_lat_i = min(H, lat_idx + 5)
    min_lon_i = max(0, lon_idx - 4)
    max_lon_i = min(W, lon_idx + 5)
    
    # Grid coordinates corresponding to cropped slices
    lons_sub = np.linspace(67.5, 97.5, W)[min_lon_i:max_lon_i]
    lats_sub = np.linspace(7.5, 37.5, H)[min_lat_i:max_lat_i]
    
    rain_sub = data[-1, 0, min_lat_i:max_lat_i, min_lon_i:max_lon_i] if data is not None else np.zeros((9, 9))
    tmax_sub = data[-1, 1, min_lat_i:max_lat_i, min_lon_i:max_lon_i] if data is not None else np.zeros((9, 9))
    
    col_map1, col_map2 = st.columns(2)
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    with col_map1:
        st.markdown("##### 🌡️ Regional Temperature Grid (°C)")
        fig_map_temp = px.imshow(
            tmax_sub,
            labels=dict(x="Longitude", y="Latitude", color="Temp (°C)"),
            x=lons_sub,
            y=lats_sub,
            color_continuous_scale="Viridis",
            origin="lower"
        )
        fig_map_temp.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=340
        )
        fig_map_temp.add_trace(go.Scatter(
            x=[lon], y=[lat], mode="markers+text",
            marker=dict(size=14, color="red", symbol="star"),
            text=[city_name.split()[0]], textposition="top center",
            name="Selected Location"
        ))
        st.plotly_chart(fig_map_temp, use_container_width=True)

    with col_map2:
        st.markdown("##### 🌧️ Regional Rainfall Grid (mm)")
        fig_map_rain = px.imshow(
            rain_sub,
            labels=dict(x="Longitude", y="Latitude", color="Rain (mm)"),
            x=lons_sub,
            y=lats_sub,
            color_continuous_scale="Cividis",
            origin="lower"
        )
        fig_map_rain.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=340
        )
        fig_map_rain.add_trace(go.Scatter(
            x=[lon], y=[lat], mode="markers+text",
            marker=dict(size=14, color="cyan", symbol="star"),
            text=[city_name.split()[0]], textposition="top center",
            name="Selected Location"
        ))
        st.plotly_chart(fig_map_rain, use_container_width=True)

    # Multi-Metric Minimal Plotly Charts
    st.markdown("---")
    st.subheader(f"📈 {forecast_days}-Day Meteorological Analytics")
    
    dates_arr = [item['date'] for item in forecast_list]
    
    tab_temp, tab_rain, tab_wind_rh, tab_uv_dew = st.tabs([
        "🌡️ Temperature Horizon", 
        "🌧️ Precipitation & Chance", 
        "💨 Wind & Humidity", 
        "☀️ UV & Dew Point"
    ])
    
    with tab_temp:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=dates_arr, y=[item['tmax'] for item in forecast_list], mode='lines+markers', name='Max Temp (°C)', line=dict(color='#FF6B6B', width=2)))
        fig_temp.add_trace(go.Scatter(x=dates_arr, y=[item['tmin'] for item in forecast_list], mode='lines+markers', name='Min Temp (°C)', line=dict(color='#4D96FF', width=2)))
        fig_temp.add_trace(go.Scatter(x=dates_arr, y=[item['feels_like'] for item in forecast_list], mode='lines', name='Feels Like (°C)', line=dict(color='#FFD93D', dash='dot')))
        fig_temp.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
    with tab_rain:
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(x=dates_arr, y=[item['rain'] for item in forecast_list], name='Rainfall (mm)', marker_color='#38BDF8', opacity=0.8))
        fig_rain.add_trace(go.Scatter(x=dates_arr, y=[item['rain_prob'] for item in forecast_list], name='Rain Chance (%)', yaxis="y2", mode='lines+markers', line=dict(color='#0284C7', width=2)))
        fig_rain.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Probability (%)", overlaying="y", side="right", range=[0, 100]),
            hovermode="x unified",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        
    with tab_wind_rh:
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(x=dates_arr, y=[item['rh'] for item in forecast_list], name='Humidity (%)', mode='lines+markers', line=dict(color='#34D399', width=2)))
        fig_wind.add_trace(go.Scatter(x=dates_arr, y=[item['wind_speed'] for item in forecast_list], name='Wind Speed (km/h)', yaxis="y2", mode='lines+markers', line=dict(color='#F472B6', width=2)))
        fig_wind.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis=dict(title="Humidity (%)"),
            yaxis2=dict(title="Wind Speed (km/h)", overlaying="y", side="right"),
            hovermode="x unified",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_wind, use_container_width=True)

    with tab_uv_dew:
        fig_uv = go.Figure()
        fig_uv.add_trace(go.Bar(x=dates_arr, y=[item['uv'] for item in forecast_list], name='UV Index', marker_color='#F59E0B', opacity=0.85))
        fig_uv.add_trace(go.Scatter(x=dates_arr, y=[item['dew_point'] for item in forecast_list], name='Dew Point (°C)', yaxis="y2", mode='lines+markers', line=dict(color='#10B981', width=2)))
        fig_uv.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis=dict(title="UV Index", range=[0, 12]),
            yaxis2=dict(title="Dew Point (°C)", overlaying="y", side="right"),
            hovermode="x unified",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_uv, use_container_width=True)

    # Full Data Table Export
    st.markdown("---")
    st.subheader("📋 Forecast Dataset & Download")
    df_export = pd.DataFrame([{
        "Date": item["date"].strftime("%Y-%m-%d"),
        "Day": item["date"].strftime("%A"),
        "Condition": item["condition"],
        "Max Temp (°C)": f"{item['tmax']:.1f}",
        "Min Temp (°C)": f"{item['tmin']:.1f}",
        "Feels Like (°C)": f"{item['feels_like']:.1f}",
        "Rainfall (mm)": f"{item['rain']:.1f}",
        "Rain Chance (%)": f"{item['rain_prob']}%",
        "Humidity (%)": f"{item['rh']:.0f}%",
        "Wind Speed (km/h)": f"{item['wind_speed']:.1f}",
        "Wind Dir": item["wind_dir"],
        "UV Index": f"{item['uv']:.0f}"
    } for item in forecast_list])
    
    col_tbl, col_dl = st.columns([4, 1])
    with col_dl:
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"{city_name.split()[0]}_20_day_weather_forecast.csv",
            mime="text/csv"
        )
    
    st.dataframe(df_export, use_container_width=True)

if __name__ == "__main__":
    main()
