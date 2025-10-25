import numpy as np
import torch
from torch.utils.data import Dataset


class PreEncodedDataset(Dataset):
    """Custom Dataset for pre-encoded text embeddings and categorical features."""

    def __init__(
        self, text_embeddings: np.ndarray, categorical: np.ndarray, targets: np.ndarray
    ) -> None:
        """Initializes the PreEncodedDataset.

        Args:
            text_embeddings: np.ndarray, Pre-encoded text embeddings.
            categorical: np.ndarray, Categorical features.
            targets: np.ndarray, Target values.
        """
        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        self.categorical = torch.tensor(categorical, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample by its index.

            Args:
                idx: int, The index of the sample to retrieve.
            Returns:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                    - text_embeddings: torch.Tensor, The text embeddings.
                    - categorical: torch.Tensor, The categorical features.
                    - target: torch.Tensor, The target value.
        """
        return (self.text_embeddings[idx], self.categorical[idx], self.targets[idx])
