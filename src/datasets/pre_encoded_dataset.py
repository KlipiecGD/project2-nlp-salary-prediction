import numpy as np
import torch
from torch.utils.data import Dataset


class PreEncodedDataset(Dataset):
    """Custom Dataset for pre-encoded text embeddings and categorical features."""

    def __init__(
        self, text_embeddings: np.ndarray, categorical: np.ndarray, targets: np.ndarray
    ):
        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        self.categorical = torch.tensor(categorical, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.text_embeddings[idx], self.categorical[idx], self.targets[idx])
