from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CITIES = {
    "london": (51.51, -0.13),
    "birmingham": (52.49, -1.89),
    "manchester": (53.48, -2.24),
    "leeds": (53.80, -1.55),
    "glasgow": (55.86, -4.25),
    "liverpool": (53.41, -2.99),
    "newcastle": (54.98, -1.62),
}

# Approximate population-based weights
CITY_WEIGHTS = {
    "london": 0.47,
    "birmingham": 0.15,
    "manchester": 0.14,
    "leeds": 0.10,
    "glasgow": 0.06,
    "liverpool": 0.05,
    "newcastle": 0.04,
}

YEARS = range(2009, 2025)


def load_city_temperature(city: str, lat: float, lon: float) -> pd.Series:
    """Load and concatenate all yearly .nc files for one city into a single
    hourly temperature series (in Celsius), extracted at the nearest grid point."""
    yearly_series = []

    for year in YEARS:
        path = RAW_DIR / f"era5_{city}_{year}.nc"
        if not path.exists():
            print(f"Missing file, skipping: {path}")
            continue

        ds = xr.open_dataset(path)

        # Select nearest grid point to the city's exact coordinates
        point = ds["t2m"].sel(latitude=lat, longitude=lon, method="nearest")

        # If multiple grid points came back (e.g. from a slightly larger box), average them
        if "latitude" in point.dims or "longitude" in point.dims:
            point = point.mean(dim=[d for d in ["latitude", "longitude"] if d in point.dims])

        series = point.to_series()
        series.index.name = "datetime"
        yearly_series.append(series)
        ds.close()

    if not yearly_series:
        raise FileNotFoundError(f"No downloaded files found for {city}")

    combined = pd.concat(yearly_series).sort_index()
    combined = combined - 273.15
    combined.name = city

    return combined


def build_national_temperature() -> pd.Series:
    """Combine all cities into a single population-weighted national temperature series."""
    city_series = {}

    for city, (lat, lon) in CITIES.items():
        print(f"Loading {city}...")
        city_series[city] = load_city_temperature(city, lat, lon)

    df = pd.concat(city_series, axis=1)

    # Resample hourly -> half-hourly via linear interpolation, to match demand data resolution
    df = df.resample("30min").interpolate(method="linear")

    # Sanity check: values should be in a plausible UK range
    if df.min().min() < -25 or df.max().max() > 40:
        print("Warning: temperature values outside expected UK range — check units/extraction.")

    # Population-weighted average across cities
    weights = pd.Series(CITY_WEIGHTS)
    national_temp = (df * weights).sum(axis=1) / weights.sum()
    national_temp.name = "temperature"

    return national_temp


def merge_with_demand(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the national weighted temperature series into the main demand dataframe,
    aligned on the datetime index."""
    national_temp = build_national_temperature()
    merged = demand_df.join(national_temp, how="left")

    missing = merged["temperature"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows have no matching temperature value after merge.")

    return merged


if __name__ == "__main__":
    from energy_forecast.data.load import load_raw_data
    from energy_forecast.data.preprocess import clean_data

    demand_df = clean_data(load_raw_data())
    merged_df = merge_with_demand(demand_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "demand_with_temperature.csv"
    merged_df.to_csv(output_path)
    print(f"Saved merged dataset to {output_path}")