import matplotlib.pyplot as plt
import pandas as pd

def spot_perpetual_trend(df_spot: pd.DataFrame, df_perp: pd.DataFrame):
    plt.figure(figsize=(14,6))
    plt.plot(df_spot.index, df_spot["close"], label="BTC Spot", alpha=0.7)
    plt.plot(df_perp.index, df_perp["close"], label="BTC Perpetual", alpha=0.7)
    plt.title("BTC Spot vs BTCUSDT Perpetual - Close Price Trend")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.legend()
    plt.show()

# TODO: plot out the different graphs for varying z_thr, exit_value, window
def visualise_parameters():
    pass