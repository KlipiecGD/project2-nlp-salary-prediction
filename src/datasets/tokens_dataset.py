import numpy as np
import torch
from torch.utils.data import Dataset


class TokensDataset(Dataset):
    """
    Custom PyTorch Dataset for handling multiple inputs of text indices, categorical features, and targets.
    """

    def __init__(
        self, text_indices: np.ndarray, categorical: np.ndarray, targets: np.ndarray
    ):
        self.text_idx = torch.tensor(text_indices, dtype=torch.long)
        self.categorical = torch.tensor(categorical, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.text_idx[idx], self.categorical[idx], self.targets[idx])
