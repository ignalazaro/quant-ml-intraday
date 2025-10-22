from pathlib import Path
from typing import Literal

import pandas as pd


def load_ohlc_csv(
    path: str | Path,
    timeframe: Literal["1h", "30m"] = "1h",
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Carga un CSV con columnas: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    Devuelve un DataFrame indexado por datetime.

    Args:
        path: ruta al CSV local.
        timeframe: timeframe esperado (solo referencia).
        tz: zona horaria a aplicar.

    Returns:
        DataFrame con columnas normalizadas y datetime indexado.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError("El CSV debe contener columna 'timestamp'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.tz_convert(tz)

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")

    return df
