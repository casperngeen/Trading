import numpy as np
import pandas as pd

def backtest_minute(
    df: pd.DataFrame, signal: pd.Series, fee_bps: float = 10.0
) -> pd.DataFrame:
    # Log returns per bar
    ret = np.log(df["close"]).diff().fillna(0.0)
    # Execute next bar: use position decided at t at t+1
    pos = signal.shift(1).fillna(0).astype(int)
    pnl = pos * ret
    # Fee on position changes (entry/exit). Counts flips as two changes.
    changes = np.abs(pos - pos.shift(1)).fillna(np.abs(pos))
    fee = (fee_bps / 1e4) * changes
    pnl_after_fee = pnl - fee
    out = pd.DataFrame(
        {"ret_log": ret, 
         "position": pos, 
         "changes": changes, 
         "pnl_log": pnl,
         "pnl_after_fee": pnl_after_fee
        },
        index=df.index,
    )
    return out

def forward_test(df: pd.DataFrame, fee_bps: float = 10.0):
    pass

def perf_summary(pnl_log: pd.Series) -> dict:
    pnl_log = pnl_log.dropna()
    if pnl_log.std() == 0:
        return {"sharpe_ann": 0.0, "cum_return": 0.0, "max_dd": 0.0}
    ann_factor = np.sqrt(525_600)  # minutes/year
    sharpe = pnl_log.mean() / pnl_log.std() * ann_factor
    equity = np.exp(pnl_log.cumsum())
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return {
        "sharpe_ann": float(sharpe),
        "cum_return": float(equity.iloc[-1] - 1.0),
        "max_dd": float(dd.min()),
    }