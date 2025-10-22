import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Simula ejecución bar-to-bar: decisión en t → PnL en t+1.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        return_col: str = "target",
        fee: float = 0.0005,
        slippage: float = 0.0002,
    ) -> None:
        self.df = df.copy()
        self.signal_col = signal_col
        self.return_col = return_col
        self.fee = fee
        self.slippage = slippage

    def run(self) -> pd.DataFrame:
        df = self.df.copy()
        # Retorno efectivo t→t+1
        df["gross_ret"] = df[self.return_col]
        df["trade_ret"] = df[self.signal_col].shift(1) * df["gross_ret"]
        df["cost"] = np.abs(df[self.signal_col].diff().fillna(0)) * (self.fee + self.slippage)
        df["net_ret"] = df["trade_ret"] - df["cost"]
        df["cum_ret"] = (1 + df["net_ret"]).cumprod()
        return df
