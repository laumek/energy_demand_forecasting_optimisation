import time
from pathlib import Path
import cdsapi

client = cdsapi.Client()

cities = {
    "london": (51.51, -0.13),
    "birmingham": (52.49, -1.89),
    "manchester": (53.48, -2.24),
    "leeds": (53.80, -1.55),
    "glasgow": (55.86, -4.25),
    "liverpool": (53.41, -2.99),
    "newcastle": (54.98, -1.62),
}

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30

failed_downloads = []

def download_one(city, lat, lon, year):
    target = OUTPUT_DIR / f"era5_{city}_{year}.nc"

    if target.exists():
        print(f"Skipping {city} {year} (already downloaded)")
        return

    request = {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [lat + 0.1, lon - 0.1, lat - 0.1, lon + 0.1],
        "format": "netcdf",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve("reanalysis-era5-single-levels", request, str(target))
            print(f"Downloaded {city} {year}")
            return
        except Exception as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed for {city} {year}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    failed_downloads.append((city, year))


for city, (lat, lon) in cities.items():
    for year in range(2009, 2025):
        download_one(city, lat, lon, year)

if failed_downloads:
    print(f"\n{len(failed_downloads)} downloads failed after {MAX_RETRIES} attempts each:")
    for city, year in failed_downloads:
        print(f"  - {city} {year}")
else:
    print("\nAll downloads completed successfully.")