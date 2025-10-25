import numpy as np
import torch
from torch.utils.data import Dataset


class TokensDataset(Dataset):
    """
    Custom PyTorch Dataset for handling multiple inputs of text indices, categorical features, and targets.
    """

    def __init__(
        self, text_indices: np.ndarray, categorical: np.ndarray, targets: np.ndarray
    ) -> None:
        """Initializes the TokensDataset.

        Args:
            text_indices: np.ndarray, Indices of text tokens.
            categorical: np.ndarray, Categorical features.
            targets: np.ndarray, Target values.
        """
        self.text_idx = torch.tensor(text_indices, dtype=torch.long)
        self.categorical = torch.tensor(categorical, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a single sample by its index.

        Args:
            idx: int, The index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - text_indices: torch.Tensor, The text token indices.
                - categorical: torch.Tensor, The categorical features.
                - target: torch.Tensor, The target value.
        """
        return (self.text_idx[idx], self.categorical[idx], self.targets[idx])
