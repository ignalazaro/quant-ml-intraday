# Quant ML Intraday

Estrategia intradía cuantitativa basada en Machine Learning (PyTorch), con backtesting bar-to-bar y validación walk-forward.

```mermaid
flowchart TD
    A[Ingesta OHLC] --> B[Features]
    B --> C[Entrenamiento (Linear/MLP)]
    C --> D[Backtest t→t+1]
    D --> E[Métricas & Reportes]
