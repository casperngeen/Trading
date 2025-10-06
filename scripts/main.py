import pandas as pd
import matplotlib.pyplot as plt
from util import backtest_minute, perf_summary
from plot import spot_perpetual_trend

def mean_reversion_signal(
    close: pd.Series, window: int = 50, z_thr: float = 2.0, exit_value: float = -0.5
) -> pd.Series:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - ma) / (std + 1e-12)
    # Use only info up to previous bar to avoid lookahead
    z_lag = z.shift(1)
    sig = pd.Series(0, index=close.index, dtype=int)

    sig = pd.Series(0, index=close.index, dtype=int)
    position = 0
    for t in range(len(close)):
        if position == 0: # currently not in position
            if z_lag.iloc[t] < -z_thr:
                position = 1   # enter long
        elif position == 1:  # holding long
            if z_lag.iloc[t] >= exit_value:
                position = 0   # exit back to flat
        sig.iloc[t] = position
    return sig, z

"""
Rule-base mean reversion
Based on: 1) Window length of Min 2) Entry Z-Thr 3) Exit Z-Thr
Doesn't seem to work well for minute-level trading freq (or doesnt work well for BTC)
"""
def mean_reversion():
    df = pd.read_parquet("data/processed/BTCUSDT-spot-1min-22-24.parquet").set_index("timestamp")
    window = 240
    z_thr = 2
    exit_value = 0
    signal, _ = mean_reversion_signal(df["close"], window=window, z_thr=z_thr, exit_value=exit_value)
    bt = backtest_minute(df, signal, fee_bps=10.0)
    print(perf_summary(bt["pnl_after_fee"]))

def compare_perp_spot():
    df_spot = pd.read_parquet("data/processed/BTCUSDT-1min-year-to-date.parquet").set_index("timestamp")
    df_perp = pd.read_parquet("data/processed/BTCUSDT-perp-1min-year-to-date.parquet").set_index("timestamp")
    spot_perpetual_trend(df_spot=df_spot.loc["2025-05-01"], df_perp=df_perp.loc["2025-05-01"])

if __name__ == "__main__":
    mean_reversion()
    # compare_perp_spot()