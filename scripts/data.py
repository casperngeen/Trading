import os
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import requests
import zipfile
import pandas as pd

def download_data():
    # --- PARAMETERS ---
    symbol = "BTCUSDT"
    interval = "1m"
    save_folder = "data/raw"
    merged_file_csv = "data/processed/BTCUSDT-spot-1min-22-24.csv"
    merged_file_parquet = "data/processed/BTCUSDT-spot-1min-22-24.parquet"
    product = {
        "perp": "futures/um",
        "spot": "spot"
    }
    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    today = datetime.now(timezone.utc)
    start_date = datetime(year=2025, month=1, day=1, tzinfo=timezone.utc)

    dates = []
    current_date = start_date

    for year in range(2022, 2025):
        for month in range(1,13):
            dates.append((year, month))

    base_url = "https://data.binance.vision/data/{productUrl}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"

    all_dfs = []

    for year, month in dates:
        # Download file if not already present
        productKey = "spot"
        url = base_url.format(productUrl=product[productKey], symbol=symbol, interval=interval, year=year, month=month)
        zip_path = os.path.join(save_folder, f"{productKey}-{symbol}-{interval}-{year}-{month:02d}.zip")
        csv_path = f"{symbol}-{interval}-{year}-{month:02d}.csv"

        if not os.path.exists(zip_path):
            print(f"Downloading {url} ...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                print(f"Saved: {zip_path}")
            else:
                print(f"Failed to download: {url} (status {response.status_code})")
                continue
        else:
            print(f"{zip_path} already exists.")

        # Extract CSV and read as DataFrame
        with zipfile.ZipFile(zip_path) as z:
            with z.open(csv_path) as f:
                df = pd.read_csv(f, header=None, names=cols)
                all_dfs.append(df)

    # --- Concatenate all months ---
    full_df = pd.concat(all_dfs, ignore_index=True)

    # full_df.rename(columns={"open_time": "timestamp"}, inplace=True)
    # Convert timestamp column to readable format (microsecond -> us, milliseconds -> ms)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], unit="ms")
    # full_df = full_df.sort_values("timestamp")

    # Keep only main columns for trading
    full_df = full_df[["timestamp", "open", "high", "low", "close", "volume"]]

    print("Total rows:", len(full_df))
    print(full_df.head())

    # Save to disk
    full_df.to_csv(merged_file_csv, index=False)
    full_df.to_parquet(merged_file_parquet)

    print(f"Saved merged CSV: {merged_file_csv}")
    print(f"Saved merged Parquet: {merged_file_parquet}")

'''
NOT USED for now
Used to change kline from 1 minute data to k-minute data
'''
# def resample_data(k: int, original_csv_file, new_file):
#     df = pd.read_csv(original_csv_file)
#     df = df.set_index("timestamp").sort_index()
#     df.index = pd.to_datetime(df.index)
#     df_km = df.resample(f"{k}min").agg({
#         "open": "first",
#         "high": "max",
#         "low":  "min",
#         "close":"last",
#         "volume":"sum"
#     }).dropna()

#     df_km.to_csv(f"{new_file}.csv", index=True)
#     df_km.reset_index().to_parquet(f"{new_file}.parquet", index=False)
#     print("Saved resampled files to disk")

if __name__ == "__main__":
    download_data()
    # resample_data(5, "data/processed/BTCUSDT-1min-year-to-date.csv", "data/processed/BTCUSDT-5min-year-to-date")