import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Union


class SalaryDataset(Dataset):
    def __init__(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> None:
        """
        Custom PyTorch Dataset for salary prediction using given features.

        This class handles the conversion of preprocessed features and scaled targets
        from numpy arrays or pandas data structures into PyTorch tensors.
        Prepared to use both only with categorical features and with mixed features.

        Args:
            X: Union[np.ndarray, pd.DataFrame], Features for the dataset, can be a numpy array or a pandas DataFrame.
            y: Union[np.ndarray, pd.Series], Target values (salaries) for the dataset, can be a numpy array or a pandas Series.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample by its index.

            Args:
                idx: int, The index of the sample to retrieve.
            Returns:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - features: torch.Tensor, The input features.
                    - target: torch.Tensor, The target value.
        """
        return self.X[idx], self.y[idx]
