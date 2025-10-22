from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quantml.io.loaders import load_ohlc_csv
from quantml.features.ohlc import build_features
from quantml.modeling.datasets import OhlcDataset
from quantml.modeling.models import LinearRegressor, MLPRegressor
from quantml.modeling.train import train_epoch, validate
from quantml.backtest.engine import BacktestEngine
from quantml.backtest.evaluation import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_cagr,
)
from quantml.utils.config import read_yaml
from quantml.utils.logging import get_logger


def _ensure_reports_dir(base: Path) -> Path:
    out = base / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _pick_features(df: pd.DataFrame) -> list[str]:
    exclude = {"open", "high", "low", "close", "volume", "gross_ret"}
    exclude.update({"target", "signal", "trade_ret", "net_ret", "cum_ret"})
    feats: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    if not feats:
        raise ValueError("No feature columns found. Did you run build_features?")
    return feats


def _gen_synth_csv(csv_path: Path, n: int = 500, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0.0002, 0.003, n)
    close = 100 * (1 + pd.Series(ret, index=ts)).cumprod()
    high = close * (1 + rng.normal(0.0005, 0.0008, n))
    low = close * (1 - rng.normal(0.0005, 0.0008, n)).clip(min=0)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = rng.integers(1000, 5000, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": vol,
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _make_model(model_name: str, in_dim: int, cfg: dict) -> torch.nn.Module:
    name = model_name.lower()
    if name == "linear":
        return LinearRegressor(in_dim)
    if name == "mlp":
        mlp_cfg = cfg.get("mlp", {})
        hidden_sizes: Sequence[int] = mlp_cfg.get("hidden_sizes", [64, 32])
        dropout: float = float(mlp_cfg.get("dropout", 0.0))
        hidden = int(hidden_sizes[0]) if hidden_sizes else 32
        return MLPRegressor(in_dim, hidden_dim=hidden, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    logger = get_logger()
    root = Path(".").resolve()

    # === Configuración ===
    cfg_data = read_yaml(root / "configs" / "data.yaml")
    cfg_feat = read_yaml(root / "configs" / "features.yaml")
    cfg_model = read_yaml(root / "configs" / "model.yaml")
    cfg_bt = read_yaml(root / "configs" / "backtest.yaml")

    data_dir = Path(cfg_data.get("data_dir", "data"))
    tz = cfg_data.get("tz", "UTC")
    timeframe = cfg_data.get("timeframe", "1h")
    symbol = (cfg_data.get("symbols") or ["SYNTH"])[0]
    csv_path = data_dir / "processed" / f"{symbol}_{timeframe}.csv"

    # === Datos ===
    if not csv_path.exists():
        logger.info(f"No se encontró CSV en {csv_path}. Generando datos sintéticos…")
        _gen_synth_csv(csv_path)

    logger.info(f"Cargando OHLC desde {csv_path}")
    ohlc = load_ohlc_csv(csv_path, timeframe=timeframe, tz=tz)

    ohlc["target"] = ohlc["close"].shift(-1) / ohlc["close"] - 1.0
    logger.info("Construyendo features…")
    feats = build_features(ohlc, cfg_feat)
    feats["target"] = ohlc["target"].reindex(feats.index)
    feats = feats.dropna()

    feature_cols = _pick_features(feats)

    batch_size = int(cfg_model.get("mlp", {}).get("batch_size", 256))
    epochs = int(cfg_model.get("mlp", {}).get("epochs", 20))
    lr = float(cfg_model.get("mlp", {}).get("lr", 1e-3))
    model_name = cfg_model.get("model", "mlp")

    # === Walk-Forward Rolling Training ===
    logger.info(
        f"Iniciando walk-forward: train={cfg_bt['walkforward']['train_bars']}, "
        f"test={cfg_bt['walkforward']['test_bars']}"
    )
    wf_train = int(cfg_bt["walkforward"]["train_bars"])
    wf_test = int(cfg_bt["walkforward"]["test_bars"])

    preds_all = np.zeros(len(feats)) * np.nan
    for start in range(0, len(feats) - wf_train - wf_test, wf_test):
        end_train = start + wf_train
        end_test = end_train + wf_test
        train_df = feats.iloc[start:end_train]
        test_df = feats.iloc[end_train:end_test]

        train_ds = OhlcDataset(train_df, feature_cols, target_col="target")
        test_ds = OhlcDataset(test_df, feature_cols, target_col="target")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = _make_model(model_name, in_dim=len(feature_cols), cfg=cfg_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = validate(model, test_loader, loss_fn)
        logger.info(f"WF [{start}:{end_test}] val_loss={val_loss:.6f}")

        with torch.no_grad():
            X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
            preds_all[end_train:end_test] = model(X_test).squeeze(1).numpy()

    feats["signal"] = np.nan_to_num(np.sign(preds_all), nan=0.0)


    # === Backtest ===
    fee = float(cfg_bt.get("fees", 0.0005))
    if "fees_bps" in cfg_bt:
        fee = float(cfg_bt["fees_bps"]) / 10000.0
    slippage = float(cfg_bt.get("slippage", 0.0002))
    if "slippage_bps" in cfg_bt:
        slippage = float(cfg_bt["slippage_bps"]) / 10000.0

    engine = BacktestEngine(
        feats, signal_col="signal", return_col="target", fee=fee, slippage=slippage
    )
    bt = engine.run()

    # === Métricas ampliadas ===
    periods_per_year = int(cfg_bt.get("periods_per_year", 252))
    sharpe = compute_sharpe(bt["net_ret"], periods_per_year)
    sortino = compute_sortino(bt["net_ret"], periods_per_year)
    maxdd = compute_max_drawdown(bt["cum_ret"])
    cagr = compute_cagr(bt["cum_ret"], periods_per_year, freq=timeframe)
    rolling_sharpe = (
        bt["net_ret"]
        .rolling(50)
        .apply(lambda x: compute_sharpe(x, periods_per_year=periods_per_year))
    )

    # === Reportes ===
    reports = _ensure_reports_dir(root)
    bt["drawdown"] = bt["cum_ret"] / bt["cum_ret"].cummax() - 1
    bt.to_csv(reports / "backtest_results.csv", index=True)

    # Equity
    plt.figure(figsize=(8, 4))
    bt["cum_ret"].plot(title="Equity Curve")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(reports / "equity_curve.png")
    plt.close()

    # Histograma
    plt.figure(figsize=(6, 4))
    bt["net_ret"].hist(bins=40)
    plt.title("Distribution of Returns")
    plt.tight_layout()
    plt.savefig(reports / "returns_hist.png")
    plt.close()

    # Rolling Sharpe
    plt.figure(figsize=(8, 4))
    rolling_sharpe.plot(title="Rolling Sharpe (50 bars)")
    plt.tight_layout()
    plt.savefig(reports / "rolling_sharpe.png")
    plt.close()

    # Drawdown
    plt.figure(figsize=(8, 4))
    bt["drawdown"].plot(title="Drawdown Curve")
    plt.tight_layout()
    plt.savefig(reports / "drawdown_curve.png")
    plt.close()

    # Markdown summary
    md_path = reports / "summary.md"
    md_path.write_text(
        f"""# QuantML Intraday Report

**Symbol:** {symbol}  
**Timeframe:** {timeframe}  

| Metric | Value |
|---------|--------|
| Sharpe | {sharpe:.2f} |
| Sortino | {sortino:.2f} |
| Max Drawdown | {maxdd:.2%} |
| CAGR | {cagr:.2%} |

---

### Gráficos
- ![Equity Curve](equity_curve.png)
- ![Drawdown](drawdown_curve.png)
- ![Rolling Sharpe](rolling_sharpe.png)
- ![Returns Histogram](returns_hist.png)
""",
        encoding="utf-8",
    )

    logger.info(f"Walk-forward completo. Reportes en {reports}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant ML Intraday — Research pipeline")
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help=(
            "Ruta a CSV OHLC con columnas timestamp, open, high, low, close, volume. "
            "Si no se indica y no existe el CSV configurado, se genera sintético."
        ),
    )
    args = parser.parse_args()

    if args.csv:
        from shutil import copyfile

        dst = Path("data/processed/custom_1h.csv")
        dst.parent.mkdir(parents=True, exist_ok=True)
        copyfile(args.csv, dst)

    main()
