"""
TODO: Try using time series cross validation to train a model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from util import backtest_minute, perf_summary

'''
H: the horizon (in minutes)
'''
def generate_features(df: pd.DataFrame, H: int = 5):
    df = df.copy()
    # drop meaningless data where there are no price changes / trading volume
    mask_const_price = (
        (df['open'] == df['high']) &
        (df['open'] == df['low']) &
        (df['open'] == df['close'])
    )
    mask_zero_vol = (df['volume'] == 0)
    mask_drop = mask_const_price & mask_zero_vol
    df = df.loc[~mask_drop].copy()
    
    df['logp'] = np.log(df['close'])
    # past log-returns
    df['ret1'] = df['logp'].diff()
    df['ret5'] = df['logp'].diff(5)
    df['ret15'] = df['logp'].diff(15)
    df['ret60'] = df['logp'].diff(60)
    df['rv10'] = df['ret1'].rolling(10).std()
    df['rv60'] = df['ret1'].rolling(60).std()
    # changes on the last hour
    df['ma60'] = df['logp'].rolling(60).mean()
    df['std60'] = df['logp'].rolling(60).std()
    df['z60'] = (df['logp'] - df['ma60']) / (df['std60'] + 1e-12)
    df['vol_chg'] = df['volume'].pct_change()
    typ = (df['high'] + df['low'] + df['close'])/3
    df['vwap_num'] = (typ * df['volume']).cumsum()
    df['vwap_den'] = df['volume'].cumsum()
    df['vwap'] = (df['vwap_num'] / df['vwap_den']).ffill()
    df['dist_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-12)
    # shift features by 1 to avoid lookahead
    feat_cols = ['rv10', 'rv60', 'z60', 'vol_chg', 'dist_vwap']
    df['ret_fwd'] = df['logp'].shift(-H) - df['logp']
    df[feat_cols] = df[feat_cols].shift(1)

    # we drop any rows with NA / inf (the first few which do not have any rolling mean)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols + ['ret_fwd']).copy()
    return df, feat_cols

'''
preds: numpy array of predicted forward returns
'''
def pred_to_signal(preds: np.array, threshold: float, pred_index: pd.Series):
    sig = np.zeros_like(preds, dtype=int)
    sig[preds > threshold] = 1
    # long-only strat for now
    # TODO: add handling for short
    # sig[preds < -threshold] = -1
    return pd.Series(sig, index=pred_index)

def train_and_eval(df_train, H=5,
                    n_splits=5, fee_bps=10.0,
                    model=None, threshold_quantile=0.9):
    df_feat, feat_cols = generate_features(df_train, H=H)
    X = df_feat[feat_cols]
    y = df_feat['ret_fwd']

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    fold_details = []

    # transformers: Scale data to Standard -> Ridge Regression
    if model is None:
        model = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]) # we can use logspace alphas here if we wanted to choose the best alpha value

    # tscv.split() gives us the relevant indices of the split data that are used for train/test at each iteration
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr) # learns the parameters in pipeline (mean/std dev in StandardScalar, regression coeff)
        pred_val = model.predict(X_val) # generate predicted Ys based on test data

        # we generate the predicted forward returns based on our training data
        pred_tr = model.predict(X_tr)
        # we find the threshold value based of some threshold quantile specified -> signal of when is a good time to move
        thr = np.quantile(np.abs(pred_tr), threshold_quantile)

        # make signals aligned with original timestamps for backtest
        pred_index = X_val.index
        signals = pred_to_signal(pred_val, threshold=thr, pred_index=pred_index)

        # backtest on the validation slice using original df slice
        df_val_full = df_train.loc[X_val.index[0]: X_val.index[-1]]  # select matching timeframe
        bt = backtest_minute(df_val_full, signals, fee_bps=fee_bps)
        perf = perf_summary(bt['pnl_after_fee'])
        fold_scores.append(perf['sharpe_ann'])
        fold_details.append({'fold': fold, 'sharpe': perf['sharpe_ann'], 'thr': thr})

    mean_sharpe = np.nanmean(fold_scores)
    return mean_sharpe, fold_details, model, feat_cols

def refit_and_test(df_train, df_test, feat_cols, model, H=5, fee_bps=10.0, threshold_quantile=0.7):
    df_train_feat, _ = generate_features(df_train, H=H)
    X_train = df_train_feat[feat_cols]
    y_train = df_train_feat['ret_fwd']
    
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    thr = np.quantile(np.abs(pred_train), threshold_quantile)

    # prepare test features
    df_test_feat, _ = generate_features(df_test, H=H)
    X_test = df_test_feat[feat_cols]
    pred_test = model.predict(X_test)

    sig_test = pred_to_signal(pred_test, threshold=thr, pred_index=X_test.index)
    bt_test = backtest_minute(df_test.loc[X_test.index[0]:X_test.index[-1]], sig_test, fee_bps=fee_bps)
    return perf_summary(bt_test['pnl_after_fee']), bt_test