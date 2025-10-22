import torch


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.mean((y_true - y_pred) ** 2).item()


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1 - ss_res / ss_tot).item()


def sharpe_ratio(returns: torch.Tensor) -> float:
    mean = torch.mean(returns)
    std = torch.std(returns)
    return (mean / std).item() if std > 0 else 0.0
