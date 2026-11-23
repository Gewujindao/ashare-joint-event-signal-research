"""Vectorized feature helpers for the A-share engine (wide [days x stocks] arrays)."""
import numpy as np
import pandas as pd


def df(x):
    return pd.DataFrame(x)


def rolling_max_prev(x, n):
    return df(x).shift(1).rolling(n, min_periods=n).max().to_numpy(np.float32)


def rolling_min_prev(x, n):
    return df(x).shift(1).rolling(n, min_periods=n).min().to_numpy(np.float32)


def rolling_mean_prev(x, n):
    return df(x).shift(1).rolling(n, min_periods=n).mean().to_numpy(np.float32)


def rolling_mean(x, n):
    return df(x).rolling(n, min_periods=n).mean().to_numpy(np.float32)


def listed_days(x):
    return np.cumsum(np.isfinite(x), axis=0)


def base_universe(p, min_days=60, min_amount=5e7):
    amt20 = rolling_mean(np.nan_to_num(p.a['amount_cny'], nan=0.0), 20)
    return (listed_days(p.a['close_raw']) >= min_days) & (amt20 >= min_amount)
