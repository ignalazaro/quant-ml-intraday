from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """
    Ejecuta una época de entrenamiento sobre el DataLoader.

    Args:
        model: modelo de PyTorch.
        loader: DataLoader de entrenamiento.
        optimizer: optimizador (Adam, SGD, etc).
        loss_fn: función de pérdida.

    Returns:
        Pérdida promedio por muestra.
    """
    model.train()
    total_loss: float = 0.0
    num_samples: int = len(loader.dataset)  # type: ignore[arg-type]

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
    """
    Evalúa el modelo sin actualizar los pesos.

    Args:
        model: modelo de PyTorch.
        loader: DataLoader de validación.
        loss_fn: función de pérdida.

    Returns:
        Pérdida promedio por muestra.
    """
    model.eval()
    total_loss: float = 0.0
    num_samples: int = len(loader.dataset)  # type: ignore[arg-type]

    for xb, yb in loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total_loss += float(loss.item()) * xb.size(0)

    return total_loss / float(num_samples)
