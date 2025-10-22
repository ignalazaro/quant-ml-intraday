import numpy as np
import pandas as pd


def compute_log_return(df: pd.DataFrame) -> pd.Series:
    """Log-return sobre precios de cierre."""
    return pd.Series(np.log(df["close"] / df["close"].shift(1)), index=df.index)


def add_lagged_returns(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"ret_lag_{lag}"] = df["log_ret"].shift(lag)
    return df


def add_momentum(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"momentum_{w}"] = df["close"] / df["close"].shift(w) - 1
    return df


def add_volatility(df: pd.DataFrame, std_window: int = 12) -> pd.DataFrame:
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
    df["ret_std"] = df["log_ret"].rolling(std_window).std()
    return df


def add_zscore(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    ma = df["close"].rolling(ma_window).mean()
    std = df["close"].rolling(ma_window).std()
    df["zscore"] = (df["close"] - ma) / std
    return df


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI clásico."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pipeline completa de features."""
    df = df.copy()
    df["log_ret"] = compute_log_return(df)
    df = add_lagged_returns(df, cfg.get("returns_lags", [1, 3, 6, 12]))
    df = add_momentum(df, cfg.get("momentum_windows", [3, 6]))
    df = add_volatility(df, cfg.get("volatility", {}).get("std_window", 12))
    df = add_zscore(df, cfg.get("zscore", {}).get("ma_window", 20))

    if cfg.get("rsi", {}).get("enabled", True):
        df["rsi"] = compute_rsi(df, cfg.get("rsi", {}).get("window", 14))

    return df.dropna()
