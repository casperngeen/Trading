# Each feature has its own function, then we can choose which we want to use
import pandas as pd
import statsmodels.api as sm
from typing import List
from enum import Enum

'''
We use z-variant and not raw values since we do not want our model to be affected by magnitude
'''
class Feature(Enum):
    BASIS_Z = "basis_z"
    BASIS_Z_DIFF = "basis_z_diff"
    BASIS_Z_ACC = "basis_z_acc"
    VOLATILITY_RATIO = "vol_ratio"
    VOLUME_RATIO = "volume_ratio"
    OFI_Z_RATIO = "ofi_z_ratio"
    OFI_Z_DIFF_RATIO = "ofi_z_diff_ratio"
    OFI_Z_ACC_RATIO = "ofi_z_acc_ratio"
    SPREAD_Z = "spread_z"
    SPREAD_Z_DIFF = "spread_z_diff"
    SPREAD_Z_ACC = "spread_z_acc"
    RETURNS = "returns"

'''
Feature Engineering Orchestration:
Takes in a list of features to be generated, with relevant window frames 
'''
def generate_features(columns: List[Feature], perp: pd.DataFrame, spot: pd.DataFrame, 
                      basis_window: int, vol_window: int, ofi_window: int, coint_window: int,
                      horizon: int):
    featureDf = pd.DataFrame()
    featureDf = generate_basis_features(columns, featureDf, perp, spot, basis_window)
    featureDf = generate_volatility_features(columns, featureDf, perp, spot, vol_window)
    featureDf = generate_volume_features(columns, featureDf, perp, spot)
    featureDf = generate_order_flow_features(columns, featureDf, perp, spot, ofi_window)
    featureDf = generate_cointegration_features(columns, featureDf, perp, spot, coint_window)
    featureDf = generate_returns(featureDf, perp, spot, horizon)
    featureDf = featureDf.dropna()
    return featureDf

'''
Basis is measure of the spread of perp and spot prices (relative to spot)
We tune window as a hyperparamater in the model training step
'''
def generate_basis_features(columns: List[Feature], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    basis = (perp["close"] - spot["close"]) / spot["close"]
    mean = basis.rolling(window).mean()
    std = basis.rolling(window).std(ddof=1)
    basis_z = (basis - mean) / (std + 1e-8)
    
    # we calculate derivatives of basis_z to find how basis is changing
    basis_z_diff = basis_z.diff()
    basis_z_acc = basis_z_diff.diff()

    if Feature.BASIS_Z in columns:
        featureDf[Feature.BASIS_Z] = basis_z
    if Feature.BASIS_Z_DIFF in columns:
        featureDf[Feature.BASIS_Z_DIFF] = basis_z_diff
    if Feature.BASIS_Z_ACC in columns:
        featureDf[Feature.BASIS_Z_ACC] = basis_z_acc

    return featureDf

'''
Volatility can be used to measure risk
Different levels of volatility can also indicate periods where different strategies are applied
'''
def generate_volatility_features(columns: List[Feature], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    if Feature.VOLATILITY_RATIO in columns:
        perp_change = perp["close"].pct_change()
        spot_change = spot["close"].pct_change()
        
        # default value of ddof is 1 -> unbiased estimate of std
        featureDf[Feature.VOLATILITY_RATIO] = (
            perp_change.rolling(window).std()
            / (spot_change.rolling(window).std() + 1e-8)
        )
    return featureDf

def generate_volume_features(columns: List[Feature], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame):
    if Feature.VOLUME_RATIO in columns:
        featureDf[Feature.VOLUME_RATIO] = perp["volume"] / spot["volume"]
    return featureDf

'''
Order flow imbalance (OFI) tells us about the pressure to sell/buy from either party
'''
def generate_order_flow_features(columns: List[Feature], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    dfs = [perp, spot]
    ofi_zs, ofi_z_diffs, ofi_z_accs = [], [], []
    
    for df in dfs:
        buy_vol  = df["taker_buy_base"]
        sell_vol = df["volume"] - df["taker_buy_base"]
        ofi = buy_vol - sell_vol
        ofi_z = (ofi - ofi.rolling(window).mean()) / (ofi.rolling(window).std() + 1e-8)
        ofi_z_diff = ofi_z.diff()
        ofi_z_acc = ofi_z_diff.diff()
        
        ofi_zs.append(ofi_z)
        ofi_z_diffs.append(ofi_z_diff)
        ofi_z_accs.append(ofi_z_acc)

    if Feature.OFI_Z_RATIO in columns:
        featureDf[Feature.OFI_Z_RATIO] = ofi_zs[0] / (ofi_zs[1] + 1e-8)
    if Feature.OFI_Z_DIFF_RATIO in columns:
        featureDf[Feature.OFI_Z_DIFF_RATIO] = ofi_z_diffs[0] / (ofi_z_diffs[1] + 1e-8)
    if Feature.OFI_Z_ACC_RATIO in columns:
        featureDf[Feature.OFI_Z_ACC_RATIO] = ofi_z_accs[0] / (ofi_z_accs[1] + 1e-8)
    return featureDf

'''
Generate the beta for the hedge ratio between spot and perp
(usually its 1, but we include it here for completeness)
'''
def __static_beta(perp: pd.DataFrame, spot: pd.DataFrame):
    y = perp["close"]
    x = sm.add_constant(spot["close"])
    model = sm.OLS(y, x).fit()
    beta = model.params["close"]
    return beta

'''
Cointegration is the idea that two price series follow 
some fixed relationship in the long run (usually given in the form of hedge ratio)
'''
def generate_cointegration_features(columns: List[Feature], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    beta = __static_beta(perp, spot)
    spread = perp["close"] - beta * spot["close"]
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std(ddof=1)
    spread_z = (spread - mean) / (std + 1e-8)
    spread_z_diff = spread_z.diff()
    spread_z_acc = spread_z_diff.diff()

    if Feature.SPREAD_Z in columns:
        featureDf[Feature.SPREAD_Z] = spread_z
    if Feature.SPREAD_Z_DIFF in columns:
        featureDf[Feature.SPREAD_Z_DIFF] = spread_z_diff
    if Feature.SPREAD_Z_ACC in columns:
        featureDf[Feature.SPREAD_Z_ACC] = spread_z_acc

    return featureDf

def generate_returns(featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, horizon: int):
    basis = (perp["close"] - spot["close"]) / spot["close"] # basis is measured in percentage of spot
    futureBasis = basis.shift(-horizon)
    featureDf[Feature.RETURNS] = futureBasis - basis
    return featureDf