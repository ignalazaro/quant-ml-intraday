# 🧱 Arquitectura del Proyecto

```mermaid
graph TD
    A[Datos OHLC] --> B[Features Engine]
    B --> C[Modelo ML (Linear/MLP)]
    C --> D[Walk-Forward Validation]
    D --> E[Backtest Engine]
    E --> F[Reporting y Metrics]
    F --> G[Documentación MkDocs]
