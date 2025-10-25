import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    """
    A simple feed-forward neural network for regression tasks.

    This model consists of two hidden layers with ReLU activation and dropout,
    followed by an output layer with a single neuron.
    """

    def __init__(
        self, input_dim: int, hidden_size: int = 64, dropout_prob: float = 0.3
    ) -> None:
        """
        Initializes the SimpleRegressor model.
        Args:
            input_dim: The number of features in the input data.
            hidden_size: The number of neurons in the first hidden layer. The second
                         hidden layer will have half this number. Defaults to 64.
            dropout_prob: The dropout probability for the dropout layers. Defaults to 0.3.
        """
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            x: torch.Tensor, Input features tensor.
            Returns:
                torch.Tensor: Predicted continuous target value.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleRegressorWithNormalization(nn.Module):
    """
    A simple feed-forward neural network for regression with Batch Normalization.

    This model includes Batch Normalization layers after each linear layer and before the
    activation function, which helps to stabilize and accelerate training.
    """

    def __init__(
        self, input_dim: int, hidden_size: int = 64, dropout_prob: float = 0.3
    ) -> None:
        """
        Initializes the SimpleRegressorWithNormalization model.

        Args:
            input_dim: The number of features in the input data.
            hidden_size: The number of neurons in the first hidden layer. The second
                         hidden layer will have half this number, defaults to 64.
            dropout_prob: The dropout probability for the dropout layers, defaults to 0.3.
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
        """Defines the forward pass of the model.
        Args:
            x: torch.Tensor, Input features tensor.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
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
