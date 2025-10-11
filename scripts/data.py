import os
import requests
import zipfile
import pandas as pd
import glob
from pathlib import Path
from datetime import date

'''
Allows us to download kline data from Binance, saving it as parquet and csv files
'''
def downloadKlines(
        symbol = "BTCUSDT",
        interval = "1m",
        save_folder = "data/raw",
        processed_folder = "data/processed",
        product = "spot",
        productUrl = "spot",
        dates=None):
    COLS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    base_url = "https://data.binance.vision/data/{productUrl}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"

    for year, month in dates:
        url = base_url.format(productUrl=productUrl, symbol=symbol, interval=interval, year=year, month=month)
        zip_path = os.path.join(save_folder, f"{product}-{symbol}-{interval}-{year}-{month:02d}.zip")
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

        df: pd.DataFrame = None
        with zipfile.ZipFile(zip_path) as z:
            with z.open(csv_path) as f:
                if product == "futures":
                    df = pd.read_csv(f, header=0, names=COLS)
                else:
                    df = pd.read_csv(f, header=None, names=COLS)

        # Convert timestamp column to readable format (microsecond -> us, milliseconds -> ms)
        if year >= 2025 and product == "spot":
            df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        # Save to disk
        path_folder = f"{processed_folder}/{product}/{symbol}/{year}"
        csv_file = f"{path_folder}/{month:02d}.csv"
        parquet_file = f"{path_folder}/{month:02d}.parquet"
        os.makedirs(path_folder, exist_ok=True)
        df.to_csv(csv_file, index=False)
        df.to_parquet(parquet_file)

        print(f"Saved merged CSV: {csv_file}")
        print(f"Saved merged Parquet: {parquet_file}")

'''
Transforms parquet into pandas df for ease of manipulation
'''
def loadParquetAsDataframe(base: Path = Path("data/processed/spot/BTCUSDT"), 
        minYear: int = 2022, minMonth: int = 1,
        maxYear: int = 2024, maxMonth: int = 12):
    files = sorted(glob.glob(str(base / "????/*.parquet")))
    
    minDate = date(minYear, minMonth, 1)
    maxDate = date(maxYear, maxMonth, 1)
    def extractFileDate(file: str):
        filePath = Path(file)
        year = int(filePath.parent.name)
        month = int(filePath.stem)
        return date(year, month, 1)
    
    filteredFiles = filter(lambda file: minDate <= extractFileDate(file) <= maxDate, files)
    
    dfs = []
    for file in filteredFiles:
        dfs.append(pd.read_parquet(file))
    
    combined_df = pd.concat(dfs).set_index("open_time")
    return combined_df

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
    dates = []
    for year in range(2025, 2026):
        for month in range(1, 10):
            dates.append((year, month))
    downloadKlines(product = "futures",productUrl = "futures/um",dates=dates)
    # resample_data(5, "data/processed/BTCUSDT-1min-year-to-date.csv", "data/processed/BTCUSDT-5min-year-to-date")