import numpy as np
import pandas as pd

from quantml.backtest.engine import BacktestEngine
from quantml.backtest.evaluation import compute_cagr, compute_max_drawdown, compute_sharpe


def test_backtest_engine_basic() -> None:
    n = 100
    df = pd.DataFrame({
        "target": np.random.normal(0, 0.001, n),
        "signal": np.random.choice([-1, 0, 1], n),
    })
    engine = BacktestEngine(df)
    result = engine.run()

    assert "net_ret" in result.columns
    assert abs(result["net_ret"].mean()) < 0.01
    assert result["cum_ret"].iloc[-1] > 0


def test_evaluation_metrics() -> None:
    ret = pd.Series(np.random.normal(0.001, 0.002, 200))
    cum = (1 + ret).cumprod()

    sharpe = compute_sharpe(ret)
    dd = compute_max_drawdown(cum)
    cagr = compute_cagr(cum)

    assert -1 < dd < 0
    assert sharpe > -10
    assert -1 < cagr < 1
