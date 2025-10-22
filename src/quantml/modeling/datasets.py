import pandas as pd
import torch
from torch.utils.data import Dataset


class OhlcDataset(Dataset):
    """Dataset PyTorch a partir de un DataFrame de features y target (siguiente retorno)."""

    def __init__(
    self,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
) -> None:

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
