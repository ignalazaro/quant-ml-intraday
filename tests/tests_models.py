import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.quantml.modeling.datasets import OhlcDataset
from src.quantml.modeling.metrics import mse, r2_score
from src.quantml.modeling.models import LinearRegressor
from src.quantml.modeling.train import train_epoch, validate


def test_linear_regressor_training() -> None:
    # Datos sintéticos
    X = torch.randn(200, 5)
    y = X[:, 0:1] * 0.5 + 0.1 * torch.randn(200, 1)

    dataset = OhlcDataset(
    pd.DataFrame(
        torch.cat([X, y], dim=1).numpy(),
        columns=[f"x{i}" for i in range(5)] + ["target"],
    ),
    feature_cols=[f"x{i}" for i in range(5)],
)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LinearRegressor(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    loss_before = validate(model, loader, loss_fn)
    for _ in range(10):
        train_epoch(model, loader, optimizer, loss_fn)
    loss_after = validate(model, loader, loss_fn)

    assert loss_after < loss_before, "El modelo no aprende"

    preds = model(dataset.features)
    assert 0 <= r2_score(dataset.targets, preds) <= 1
    assert mse(dataset.targets, preds) >= 0
