"""
data.py - IMD Gridded Meteorological Data Ingestion and Processing Pipeline
Supports downloading IMD gridded files (.grd), parsing binary data,
bilinear interpolation of temperature from 1.0 deg up to 0.25 deg native rainfall resolution,
missing value treatment (-999.0, 99.9 -> NaN), and high-performance Zarr / NetCDF persistence.
"""

import os
import glob
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import RegularGridInterpolator
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Official IMD Pune endpoints (with www fallback)
IMD_URLS = {
    'rain': ('https://imdpune.gov.in/cmpg/Griddata/rainfall.php', 'rain', 135, 129),
    'max': ('https://imdpune.gov.in/cmpg/Griddata/maxtemp.php', 'maxtemp', 31, 31),
    'min': ('https://imdpune.gov.in/cmpg/Griddata/mintemp.php', 'mintemp', 31, 31)
}

# Coordinates definitions
# Rainfall: 0.25 x 0.25 deg
RAIN_LONS = np.linspace(66.5, 100.0, 135) # 135 lons
RAIN_LATS = np.linspace(6.5, 38.5, 129)   # 129 lats

# Temperature: 1.0 x 1.0 deg
TEMP_LONS = np.linspace(67.5, 97.5, 31)   # 31 lons
TEMP_LATS = np.linspace(7.5, 37.5, 31)   # 31 lats


def is_leap_year(year: int) -> bool:
    """Return True if year is a leap year."""
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))


def download_imd_grd(year: int, var: str, out_dir: str = "data/raw") -> Optional[str]:
    """
    Downloads raw .grd file for given variable and year from IMD portal.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{var}_{year}.grd")
    if os.path.exists(filename) and os.path.getsize(filename) > 100000:
        logger.info(f"Using cached {filename} ({os.path.getsize(filename)} bytes)")
        return filename

    if var not in IMD_URLS:
        raise ValueError(f"Unknown variable {var}. Must be one of {list(IMD_URLS.keys())}")

    url, param, _, _ = IMD_URLS[var]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    logger.info(f"Downloading {var} for year {year} from {url}...")
    try:
        resp = session.post(url, data={param: str(year)}, stream=True, timeout=60)
        resp.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to download {var} for {year}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return None


def parse_imd_grd(filename: str, var: str, year: int) -> np.ndarray:
    """
    Parses binary float32 .grd file into numpy array with NaNs for undefs.
    Rainfall shape: (Time, 129, 135) [Time, Lat, Lon]
    Temperature shape: (Time, 31, 31) [Time, Lat, Lon]
    """
    ndays = 366 if is_leap_year(year) else 365
    _, _, isiz, jsiz = IMD_URLS[var] # isiz=nlon, jsiz=nlat
    expected_size = ndays * isiz * jsiz

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    data = np.fromfile(filename, dtype=np.float32)
    if len(data) != expected_size:
        # Check if padded or partial
        if len(data) == ndays * (isiz * jsiz + 2):
            data = data.reshape((ndays, isiz * jsiz + 2))[:, 1:-1].flatten()
        elif len(data) >= expected_size:
            data = data[:expected_size]
        else:
            raise ValueError(f"File {filename} size {len(data)} is less than expected {expected_size}")

    if var == 'rain':
        # IMD Rain raw binary is ordered (Time, NY=129, NX=135) -> (Time, Lat, Lon)
        arr = data.reshape((ndays, jsiz, isiz))
        # Mask UNDEF values: -999.0
        arr[arr <= -900.0] = np.nan
        arr[arr > 2000.0] = np.nan # sanity check physical max
        arr[arr < 0.0] = 0.0      # non-negative rainfall
    else:
        # Temperature is 31 x 31 (Time, NY=31, NX=31) -> (Time, Lat, Lon)
        arr = data.reshape((ndays, jsiz, isiz))
        # Mask UNDEF values: 99.9
        arr[arr >= 90.0] = np.nan
        arr[arr <= -90.0] = np.nan
        
    return arr


def regrid_temperature_to_rain_grid(temp_array: np.ndarray,
                                   source_lats: np.ndarray = TEMP_LATS,
                                   source_lons: np.ndarray = TEMP_LONS,
                                   target_lats: np.ndarray = RAIN_LATS,
                                   target_lons: np.ndarray = RAIN_LONS) -> np.ndarray:
    """
    Bilinearly interpolates temperature from 1.0 deg grid to 0.25 deg target grid.
    temp_array: (Time, 31, 31) [Time, Lat, Lon]
    Returns: (Time, 129, 135) [Time, Lat, Lon]
    """
    T, H, W = temp_array.shape
    target_H, target_W = len(target_lats), len(target_lons)
    out_arr = np.full((T, target_H, target_W), np.nan, dtype=np.float32)

    # Meshgrid of target coordinates
    mesh_lats, mesh_lons = np.meshgrid(target_lats, target_lons, indexing='ij')
    points_to_interp = np.stack([mesh_lats.ravel(), mesh_lons.ravel()], axis=-1)

    for t in range(T):
        slice_2d = temp_array[t]
        # Check if entire slice is NaN
        if np.isnan(slice_2d).all():
            continue
        
        # In case of internal NaNs, fill with nearest neighbor for valid region
        valid_mask = ~np.isnan(slice_2d)
        if not valid_mask.all():
            # Interpolate only valid domain, fill exterior with NaN
            from scipy.interpolate import griddata
            known_pts = np.argwhere(valid_mask)
            # Coordinates of valid points
            coords_valid = np.column_stack([source_lats[known_pts[:, 0]], source_lons[known_pts[:, 1]]])
            values_valid = slice_2d[valid_mask]
            
            # Linear griddata interpolation
            interp_vals = griddata(coords_valid, values_valid, points_to_interp, method='linear', fill_value=np.nan)
            out_arr[t] = interp_vals.reshape(target_H, target_W)
        else:
            interp = RegularGridInterpolator((source_lats, source_lons), slice_2d, 
                                            method='linear', bounds_error=False, fill_value=np.nan)
            out_arr[t] = interp((mesh_lats, mesh_lons)).astype(np.float32)

    return out_arr


def build_weather_tensor(year: int, raw_dir: str = "data/raw", target_res: float = 0.25) -> Optional[np.ndarray]:
    """
    Processes rain, max temp, min temp for a specific year.
    Returns: (Time, Channels=3, H, W) where Channels: 0: Rain, 1: MaxTemp, 2: MinTemp
    At target_res=0.25: H=129 (Lat), W=135 (Lon)
    At target_res=1.0: H=31, W=31 (backward compatibility)
    """
    # Verify raw files exist
    files = {}
    for var in ['rain', 'max', 'min']:
        fpath = os.path.join(raw_dir, f"{var}_{year}.grd")
        if not os.path.exists(fpath):
            fpath = download_imd_grd(year, var, raw_dir)
            if not fpath:
                logger.warning(f"Missing {var} data for {year}. Skipping year.")
                return None
        files[var] = fpath

    rain_raw = parse_imd_grd(files['rain'], 'rain', year) # (T, 129, 135)
    max_raw = parse_imd_grd(files['max'], 'max', year)   # (T, 31, 31)
    min_raw = parse_imd_grd(files['min'], 'min', year)   # (T, 31, 31)

    T = rain_raw.shape[0]

    if abs(target_res - 0.25) < 1e-4:
        # Regrid max and min temp up to 0.25 deg grid (129, 135)
        logger.info(f"Interpolating temperature to 0.25 deg native rainfall resolution for year {year}...")
        max_regrid = regrid_temperature_to_rain_grid(max_raw)
        min_regrid = regrid_temperature_to_rain_grid(min_raw)
        tensor = np.stack([rain_raw, max_regrid, min_regrid], axis=1) # (T, 3, 129, 135)
    elif abs(target_res - 1.0) < 1e-4:
        # Box average rainfall down to 31x31 (legacy 1.0 deg grid)
        rain_down = np.full((T, 31, 31), np.nan, dtype=np.float32)
        for r in range(31):
            for c in range(31):
                r_start = 4 + r * 4
                c_start = 4 + c * 4
                block = rain_raw[:, r_start:r_start+4, c_start:c_start+4]
                rain_down[:, r, c] = np.nanmean(block, axis=(1, 2))
        tensor = np.stack([rain_down, max_raw, min_raw], axis=1) # (T, 3, 31, 31)
    else:
        raise ValueError(f"Unsupported target resolution: {target_res}")

    return tensor.astype(np.float32)


def process_and_save_zarr(years: List[int],
                          zarr_path: str = "data/processed/imd_weather_grid.zarr",
                          raw_dir: str = "data/raw",
                          target_res: float = 0.25) -> zarr.Group:
    """
    Processes specified years and writes to a unified Zarr store.
    The store contains:
      - 'weather': (Total_Days, 3, H, W)
      - 'dates': string array of ISO dates 'YYYY-MM-DD'
      - 'lats': 1D latitude coordinates
      - 'lons': 1D longitude coordinates
      - attrs: resolution, channels, date ranges
    """
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    root = zarr.open_group(zarr_path, mode='w')

    all_tensors = []
    all_dates = []

    for yr in sorted(years):
        tensor = build_weather_tensor(yr, raw_dir=raw_dir, target_res=target_res)
        if tensor is None:
            continue
        all_tensors.append(tensor)
        ndays = tensor.shape[0]
        base_date = datetime(yr, 1, 1)
        yr_dates = [(base_date + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(ndays)]
        all_dates.extend(yr_dates)

    if not all_tensors:
        raise RuntimeError("No weather data processed successfully!")

    combined = np.concatenate(all_tensors, axis=0) # (Total_Days, 3, H, W)
    logger.info(f"Total processed dataset shape: {combined.shape}")

    # Write to Zarr
    chunks = (30, 3, combined.shape[2], combined.shape[3])
    ds_weather = root.create_dataset('weather', data=combined, chunks=chunks, dtype='float32')
    ds_dates = root.create_dataset('dates', data=np.array(all_dates, dtype='object'), object_codec=None)
    
    if abs(target_res - 0.25) < 1e-4:
        root.create_dataset('lats', data=RAIN_LATS, dtype='float32')
        root.create_dataset('lons', data=RAIN_LONS, dtype='float32')
    else:
        root.create_dataset('lats', data=TEMP_LATS, dtype='float32')
        root.create_dataset('lons', data=TEMP_LONS, dtype='float32')

    root.attrs['channels'] = ["rainfall_mm", "tmax_degc", "tmin_degc"]
    root.attrs['target_resolution'] = target_res
    root.attrs['start_date'] = all_dates[0]
    root.attrs['end_date'] = all_dates[-1]

    logger.info(f"Successfully saved Zarr store at {zarr_path}")
    return root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IMD Gridded Data Pipeline v2.0")
    parser.add_argument("--start_year", type=int, default=2020)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--res", type=float, default=0.25)
    parser.add_argument("--zarr_path", type=str, default="data/processed/imd_weather_grid.zarr")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    process_and_save_zarr(years, zarr_path=args.zarr_path, target_res=args.res)
