from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from quantml.backtest.engine import BacktestEngine
from quantml.backtest.evaluation import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
)
from quantml.features.ohlc import build_features
from quantml.io.loaders import load_ohlc_csv
from quantml.modeling.datasets import OhlcDataset
from quantml.modeling.models import LinearRegressor, MLPRegressor
from quantml.modeling.train import train_epoch, validate
from quantml.utils.config import read_yaml
from quantml.utils.logging import get_logger


def _ensure_reports_dir(base: Path) -> Path:
    out = base / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _pick_features(df: pd.DataFrame) -> list[str]:
    # Seleccionamos todas las columnas numéricas que **no** son OHLC/base ni target
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
    # camino aleatorio suave
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
        # MLPRegressor de 2 capas (rápido). Si quieres más, amplía models.py.
        # Usamos el primer tamaño para esta versión ligera:
        hidden = int(hidden_sizes[0]) if hidden_sizes else 32
        return MLPRegressor(in_dim, hidden_dim=hidden, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    logger = get_logger()
    root = Path(".").resolve()

    # Cargar configs
    cfg_data = read_yaml(root / "configs" / "data.yaml")
    cfg_feat = read_yaml(root / "configs" / "features.yaml")
    cfg_model = read_yaml(root / "configs" / "model.yaml")
    cfg_bt = read_yaml(root / "configs" / "backtest.yaml")

    data_dir = Path(cfg_data.get("data_dir", "data"))
    tz = cfg_data.get("tz", "UTC")
    timeframe = cfg_data.get("timeframe", "1h")
    symbol = (cfg_data.get("symbols") or ["SYNTH"])[0]
    csv_path = data_dir / "processed" / f"{symbol}_{timeframe}.csv"

    # Cargar datos (si no existe, generamos sintético para poder probar hoy)
    if not csv_path.exists():
        logger.info(f"No se encontró CSV en {csv_path}. Generando datos sintéticos…")
        _gen_synth_csv(csv_path)

    logger.info(f"Cargando OHLC desde {csv_path}")
    ohlc = load_ohlc_csv(csv_path, timeframe=timeframe, tz=tz)

    # Target: retorno t->t+1
    ohlc["target"] = ohlc["close"].shift(-1) / ohlc["close"] - 1.0

    # Features
    logger.info("Construyendo features…")
    feats = build_features(ohlc, cfg_feat)
    feats["target"] = ohlc["target"].reindex(feats.index)
    feats = feats.dropna()

    feature_cols = _pick_features(feats)

    # Split simple (últimas N barras para test); WF rolling llega en sprint extra si lo deseas
    test_bars = int(cfg_bt.get("walkforward", {}).get("test_bars", 24))
    train_df = feats.iloc[:-test_bars].copy() if test_bars > 0 else feats.copy()
    test_df = feats.iloc[-test_bars:].copy() if test_bars > 0 else feats.iloc[-1:].copy()

    logger.info(
        f"Train bars: {len(train_df)} | Test bars: {len(test_df)} | Features: {len(feature_cols)}"
    )

    # Datasets / Loaders
    batch_size = int(cfg_model.get("mlp", {}).get("batch_size", 256))
    epochs = int(cfg_model.get("mlp", {}).get("epochs", 20))
    lr = float(cfg_model.get("mlp", {}).get("lr", 1e-3))

    train_ds = OhlcDataset(train_df, feature_cols, target_col="target")
    test_ds = OhlcDataset(test_df, feature_cols, target_col="target")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Modelo
    model_name = cfg_model.get("model", "mlp")
    model = _make_model(model_name, in_dim=len(feature_cols), cfg=cfg_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Entrenamiento básico
    logger.info(f"Entrenando modelo: {model_name} (epochs={epochs}, lr={lr})")
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, loss_fn)
        val = validate(model, test_loader, loss_fn)
        if val < best_val:
            best_val = val
        if epoch % max(1, epochs // 5) == 0:
            logger.info(f"Epoch {epoch:03d} | train {tr:.6f} | valid {val:.6f}")

    # Señal por signo de predicción (regresión → dirección)
    with torch.no_grad():
        X_all = torch.tensor(feats[feature_cols].values, dtype=torch.float32)
        preds = model(X_all).squeeze(1).numpy()
    feats["signal"] = np.sign(preds)

    # Backtest
    # Tolerante con configs: acepta 'fees' (fracción) o 'fees_bps' (bps)
    fee = float(cfg_bt.get("fees", 0.0005))
    if "fees_bps" in cfg_bt:
        fee = float(cfg_bt["fees_bps"]) / 10000.0
    slippage = float(cfg_bt.get("slippage", 0.0002))
    if "slippage_bps" in cfg_bt:
        slippage = float(cfg_bt["slippage_bps"]) / 10000.0

    engine = BacktestEngine(feats, signal_col="signal", return_col="target", fee=fee, slippage=slippage)
    bt = engine.run()

    # Métricas
    periods_per_year = int(cfg_bt.get("periods_per_year", 252))
    sharpe = compute_sharpe(bt["net_ret"], periods_per_year)
    sortino = compute_sortino(bt["net_ret"], periods_per_year)
    maxdd = compute_max_drawdown(bt["cum_ret"])
    cagr = compute_cagr(bt["cum_ret"], periods_per_year=periods_per_year, freq=timeframe)

    logger.info(
        f"Sharpe={sharpe:.2f} | Sortino={sortino:.2f} | MaxDD={maxdd:.2%} | CAGR={cagr:.2%}"
    )

    # Reportes
    reports = _ensure_reports_dir(root)
    eq_path = reports / "equity_curve.png"
    csv_path_out = reports / "backtest_results.csv"
    bt.to_csv(csv_path_out, index=True)

    plt.figure()
    bt["cum_ret"].plot()
    plt.title(f"Equity Curve — {symbol} ({timeframe})")
    plt.xlabel("time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(eq_path)
    plt.close()

    summary_txt = reports / "summary.txt"
    summary_txt.write_text(
        f"Symbol: {symbol}\nTimeframe: {timeframe}\n"
        f"Features: {len(feature_cols)}\n"
        f"Train bars: {len(train_df)} | Test bars: {len(test_df)}\n"
        f"Sharpe: {sharpe:.3f}\nSortino: {sortino:.3f}\nMaxDD: {maxdd:.3%}\nCAGR: {cagr:.3%}\n",
        encoding="utf-8",
    )

    logger.info(f"Reportes guardados en: {reports}")
    logger.info(f"- CSV: {csv_path_out}")
    logger.info(f"- Equity: {eq_path}")
    logger.info(f"- Summary: {summary_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant ML Intraday — Research pipeline")
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Ruta a CSV OHLC con columnas timestamp, open, high, low, close, volume. "
             "Si no se indica y no existe el CSV configurado, se genera sintético.",
    )
    args = parser.parse_args()

    # Si el usuario pasa CSV por parámetro, lo copiamos a la ruta esperada y usamos el loader estándar
    if args.csv:
        # Guardar una copia en data/processed para mantener la convención del proyecto
        from shutil import copyfile

        dst = Path("data/processed/custom_1h.csv")
        dst.parent.mkdir(parents=True, exist_ok=True)
        copyfile(args.csv, dst)

        # Ajustar configs/data.yaml para usar este CSV? No necesario: el loader abre por ruta calculada.
        # Si quieres integrarlo, sustituye manualmente en configs/data.yaml el símbolo/timeframe.

    main()
