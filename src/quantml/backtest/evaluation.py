import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    mean = returns.mean()
    std = returns.std()
    return np.sqrt(periods_per_year) * mean / std if std > 0 else 0.0


def compute_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0].std()
    mean = returns.mean()
    return np.sqrt(periods_per_year) * mean / downside if downside > 0 else 0.0


def compute_max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.expanding(min_periods=1).max()
    dd = (cum_returns / peak - 1).min()
    return dd


def compute_cagr(cum_returns: pd.Series, periods_per_year: int = 252, freq: str = "1h") -> float:
    n_periods = len(cum_returns)
    years = n_periods / periods_per_year
    total_return = cum_returns.iloc[-1]
    return total_return ** (1 / years) - 1 if years > 0 else 0.0
