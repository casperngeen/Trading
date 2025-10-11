# Each feature has its own function, then we can choose which we want to use
import pandas as pd
import statsmodels.api as sm
from typing import List

'''
Basis is measure of the spread of perp and spot prices (relative to spot)
We tune window as a hyperparamater in the model training step
'''
def generate_basis_features(columns: List[str], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    featureDf["basis"] = (perp["close"] - spot["close"]) / spot["close"]
    mean = featureDf["basis"].rolling(window).mean()
    std = featureDf["basis"].rolling(window).std(ddof=1)
    featureDf["basis_z"] = (featureDf["basis"] - mean) / (std + 1e-8)
    
    # we calculate derivatives of basis_z to find how basis is changing
    featureDf["basis_z_diff"] = featureDf["basis_z"].diff()
    featureDf["basis_z_acc"] = featureDf["basis_z_diff"].diff()

    newColumns = columns + ["basis", "basis_z", "basis_z_diff", "basis_z_acc"]
    return featureDf, newColumns

'''
Volatility can be used to measure risk
Different levels of volatility can also indicate periods where different strategies are applied
'''
def generate_volatility_features(columns: List[str], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    perp_change = perp["close"].pct_change()
    spot_change = spot["close"].pct_change()

    # default value of ddof is 1 -> unbiased estimate of std
    featureDf["vol"] = (perp_change.rolling(window).std()) / (spot_change.rolling(window).std() + 1e-8)
    
    newColumns = columns + ["vol"]
    return featureDf, newColumns

def generate_volume_features(columns: List[str], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame):
    featureDf["volume_ratio"] = perp["volume"] / spot["volume"]
    newColumns = columns + ["volume_ratio"]
    return featureDf, newColumns

'''
Order flow imbalance (OFI) tells us about the pressure to sell/buy from either party
'''
def generate_order_flow_features(columns: List[str], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    dfs = [perp, spot]
    ofis = []
    ofi_zs = []
    ofi_z_diffs = []
    ofi_z_accs = []
    
    for df in dfs:
        buy_vol  = df["taker_buy_base"]
        sell_vol = df["volume"] - df["taker_buy_base"]
        ofi = buy_vol - sell_vol
        ofi_z = (ofi - ofi.rolling(window).mean()) / (ofi.rolling(window).std() + 1e-8)
        ofi_z_diff = ofi_z.diff()
        ofi_z_acc = ofi_z_diff.diff()
        
        ofis.append(ofi)
        ofi_zs.append(ofi_z)
        ofi_z_diffs.append(ofi_z_diff)
        ofi_z_accs.append(ofi_z_acc)

    featureDf["ofi_ratio"] = ofi[0] / ofi[1] 
    featureDf["ofi_z_ratio"] = ofi_zs[0] / ofi_zs[1]
    featureDf["ofi_z_diff_ratio"] = ofi_z_diffs[0] / ofi_z_diffs[1]
    featureDf["ofi_z_acc_ratio"] = ofi_z_accs[0] / ofi_z_accs[1]
    newColumns = columns + ["ofi", "ofi_z_ratio", "ofi_z_diff_ratio", "ofi_z_acc_ratio"]
    return featureDf, newColumns

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
def generate_cointegration_features(columns: List[str], featureDf: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame, window: int):
    beta = __static_beta(perp, spot)
    featureDf["spread"] = perp["close"] - beta * spot["close"]
    
    mean = featureDf["spread"].rolling(window).mean()
    std = featureDf["spread"].rolling(window).std()
    featureDf["spread_z"] = (featureDf["spread"] - mean) / (std + 1e-8)
    
    featureDf["spread_z_diff"] = featureDf["spread_z"].diff()
    featureDf["spread_z_acc"] = featureDf["spread_z_diff"].diff()

    newColumns = columns + ["spread", "spread_z", "spread_z_diff", "spread_z_acc"]
    return featureDf, newColumns
