# 📐 Metodología

Este documento describe la estructura general del pipeline intradía basado en ML:

1. **Datos:** OHLC en 1h con features técnicas (lags, momentum, RSI, z-score).
2. **Modelo:** Linear o MLP (PyTorch) con target = retorno t→t+1.
3. **Entrenamiento:** Rolling walk-forward (train 120 barras, test 24).
4. **Backtest:** Simulación bar-to-bar con fees y slippage configurables.
5. **Métricas:** Sharpe, Sortino, MaxDD, CAGR.
