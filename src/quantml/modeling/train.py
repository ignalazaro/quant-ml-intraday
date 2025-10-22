from collections.abc import Callable  # ✅ corregido según Ruff

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.train()
    total_loss: float = 0.0
    num_samples: int = len(loader.dataset)
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / float(num_samples)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    total_loss: float = 0.0
    num_samples: int = len(loader.dataset)
    for xb, yb in loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / float(num_samples)
