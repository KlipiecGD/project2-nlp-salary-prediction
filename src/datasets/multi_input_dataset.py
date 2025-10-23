import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union, Tuple


class MultiInputDataset(Dataset):
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        tabular_start_index: int,
    ):
        """
        Custom PyTorch Dataset for handling multiple input types (e.g., embeddings and tabular data).

        This class splits the preprocessed feature matrix into its constituent parts
        (e.g., text embeddings and tabular features) and prepares them as PyTorch tensors.

        Args:
            X: The combined feature matrix from a `ColumnTransformer`, expected
               to be a numpy array or pandas DataFrame.
            y: The target values, can be a numpy array or pandas Series.
            tabular_start_index: The column index where the tabular features begin.
                                 Features before this index are considered embeddings.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.tabular_start_index = tabular_start_index

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample by its index, splitting features into embeddings and tabular data.
        """
        embeddings = self.X[idx, : self.tabular_start_index]
        tabular_features = self.X[idx, self.tabular_start_index :]

        target = self.y[idx]

        return embeddings, tabular_features, target
