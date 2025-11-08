import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import backtest_minute, perf_summary
from plot import spot_perpetual_trend
from load_data import loadParquetAsDataframe, alignData
from generate_features import generate_features, Feature
from feature_selection import correlation
from pathlib import Path
from btc_spot_perp import hyperparamTuning

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

def btc_spot_perp_model():
    # Load Data
    perp = loadParquetAsDataframe(Path("data/processed/futures/BTCUSDT"))
    spot = loadParquetAsDataframe()
    perp, spot = alignData(perp, spot)
    
    # Feature Engineering & Selection
    featureValues = list(Feature)
    # we use 100 as window size as an arbitrary gauge
    featureDf = generate_features(featureValues, perp, spot, 100, 100, 100, 100, 30)
    correlatedPairs = correlation(featureDf, 0.2)
    print(correlatedPairs)
    '''
    Upon inspection, we can see that the correlation of basis and spread is high, 
    which is due to the fact that beta is 1 -> basis and spread are linearly related
    We can also observe that the diff and acc is highly correlated, 
    and we will drop basis_z_diff since it has a higher correlation with basis_z
    '''
    # TODO: think about how to automatically select features based on correlation
    selectedFeatures = [Feature.BASIS_Z, Feature.BASIS_Z_ACC, Feature.VOLATILITY_RATIO, 
                        Feature.VOLUME_RATIO, Feature.OFI_Z_RATIO, Feature.OFI_Z_DIFF_RATIO, 
                        Feature.OFI_Z_ACC_RATIO]
    
    # Hyperparamter Tuning
    study = hyperparamTuning(selectedFeatures, perp, spot)
    print("Best Sharpe:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Best trial number:", study.best_trial.number)
    study.trials_dataframe().to_csv("optuna_results.csv")


if __name__ == "__main__":
    # mean_reversion()
    btc_spot_perp_model()
    # compare_perp_spot()