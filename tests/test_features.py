import numpy as np
import pandas as pd

from quantml.features.ohlc import build_features


def test_build_features_basic() -> None:
    # Crear datos OHLC sintéticos
    idx = pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": np.linspace(100, 150, 50),
        "high": np.linspace(101, 151, 50),
        "low": np.linspace(99, 149, 50),
        "close": np.linspace(100, 150, 50),
        "volume": np.random.randint(1000, 2000, 50),
    }).set_index("timestamp")

    cfg = {
        "returns_lags": [1, 3],
        "momentum_windows": [3],
        "volatility": {"std_window": 5},
        "zscore": {"ma_window": 5},
        "rsi": {"enabled": True, "window": 5},
    }

    features = build_features(df, cfg)
    expected_cols = [
    "log_ret",
    "ret_lag_1",
    "ret_lag_3",
    "momentum_3",
    "hl_range",
    "ret_std",
    "zscore",
    "rsi",
]


    for col in expected_cols:
        assert col in features.columns, f"Falta columna {col}"

    assert not features.isna().any().any(), "Hay NaNs inesperados"
