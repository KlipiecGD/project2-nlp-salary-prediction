import torch
import torch.nn as nn
from config.config import DROPOUT_RATE


class SimpleRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int = 64, dropout_prob: float = DROPOUT_RATE
    ):
        """
        A simple feed-forward neural network for regression tasks.

        This model consists of two hidden layers with ReLU activation and dropout,
        followed by an output layer with a single neuron.

        Args:
            input_dim: The number of features in the input data.
            hidden_size: The number of neurons in the first hidden layer. The second
                         hidden layer will have half this number. Defaults to 64.
            dropout_prob: The dropout probability for the dropout layers.
        """
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleRegressorWithNormalization(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int = 64, dropout_prob: float = DROPOUT_RATE
    ):
        """
        A simple feed-forward neural network for regression with Batch Normalization.

        This model includes Batch Normalization layers after each linear layer and before the
        activation function, which helps to stabilize and accelerate training.

        Args:
            input_dim: The number of features in the input data.
            hidden_size: The number of neurons in the first hidden layer. The second
                         hidden layer will have half this number. Defaults to 64.
            dropout_prob: The dropout probability for the dropout layers.
        """
        super(SimpleRegressorWithNormalization, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
