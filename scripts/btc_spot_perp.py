import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from generate_features import generate_features
from optuna import create_study
from optuna.trial import Trial
from typing import List
from generate_features import Feature

def sharpe_metric(model: XGBRegressor, X_val, y_val, threshold):
    preds = model.predict(X_val)
    
    # beat the cost of trade (assume worst case of spot = 0.1, taker in perp = 0.05)
    signals = np.where(preds > threshold, 1, np.where(preds < -threshold, -1, 0))
    
    FEE_BPS = 30
    fee_per_trade = FEE_BPS / 10000
    strategy_returns = y_val * signals
    trade_mask = np.abs(np.diff(signals, prepend=0)) > 0
    strategy_returns -= fee_per_trade * trade_mask.astype(float)
    
    mean_r = strategy_returns.mean()
    std_r = strategy_returns.std()
    
    if std_r == 0:
        return 0.0
    # annualized sharpe
    sharpe = (mean_r / std_r) * np.sqrt(365 * 24 * 60)
    return sharpe

def make_objective(features: List[Feature], perp: pd.DataFrame, spot: pd.DataFrame):
    def objective(trial: Trial):
        # Feature parameters
        basis_window = trial.suggest_int('basis_window', 60, 200)
        vol_window = trial.suggest_int('vol_window', 60, 200)
        ofi_window = trial.suggest_int('ofi_window', 60, 200)
        coint_window = trial.suggest_int('coint_window', 60, 200)
        horizon = trial.suggest_int('horizon', 10, 60)
        threshold = trial.suggest_float('threshold', 0.03, 0.05)
        
        # Model parameters
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)

        # Generate features with these parameters
        featureDf = generate_features(features, perp, spot, basis_window, vol_window, ofi_window, coint_window, horizon)
        print(featureDf.columns)
        X = featureDf[features]
        y = featureDf[Feature.RETURNS]
        # target = featureDf[Feature.TARGET.value]

        # Train model with PurgedCV
        scores = []
        
        purge = max(basis_window, vol_window, ofi_window, coint_window)
        embargo = horizon
        for train_idx, val_idx in purged_cv_split(featureDf, purge=purge, embargo=embargo):
            model = XGBRegressor(n_estimators=n_estimators, 
                                max_depth=6,
                                learning_rate=0.05)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(sharpe_metric(model, X.iloc[val_idx], y.iloc[val_idx], threshold))
        
        return np.mean(scores)
    return objective

def hyperparamTuning(features: List[Feature], perp: pd.DataFrame, spot: pd.DataFrame):
    study = create_study(direction='maximize')
    objective = make_objective(features, perp, spot)
    study.optimize(objective, n_trials=200)
    return study

"""
We use PurgedCV to prevent any overlap between training and testing data, eg rolling windows overlap between the two
It also prevents any delayed effects in the markets affecting the next split of data, eg unsettled PnL
purge: separation between end of train and start of test
embargo: separation between end of test and start of next test
"""
def purged_cv_split(X, n_splits=5, purge=5, embargo=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    N = len(X)
    splits = []
    
    '''
    train_idx: array of indices (of X) representing rows used for training
    test_idx: similar but used for testing
    '''
    for train_idx, test_idx in tscv.split(X):
        # apply embargo (skip a gap after previous test)
        if splits:
            last_test = splits[-1][1][-1]
            embargo_start = min(N, last_test + embargo)
            train_idx = train_idx[train_idx > embargo_start]

        # purge overlap by removing last few train_idx
        train_idx = train_idx[train_idx < test_idx[0] - purge]

        splits.append((train_idx, test_idx))
    
    return splits