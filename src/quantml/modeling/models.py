import torch
from torch import nn


class LinearRegressor(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.model = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
