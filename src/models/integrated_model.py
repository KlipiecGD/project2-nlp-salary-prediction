import torch
import torch.nn as nn


class IntegratedNN(nn.Module):
    """
    An integrated feed-forward neural network with configurable batch normalization placement.

    This model is a multi-layered perceptron designed for regression, featuring
    batch normalization and dropout for improved training stability and generalization.
    The order of batch normalization and activation can be specified.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        dropout_prob: float = 0.3,
        batch_norm_before_activation: bool = True,
    ) -> None:
        """Initializes the IntegratedNN model.

        Args:
            input_dim: int, Dimension of the input features.
            hidden_size: int, Number of neurons in the first hidden layer. Default is 256.
            dropout_prob: float, Dropout probability for regularization. Default is 0.3.
            batch_norm_before_activation: bool, If True, applies batch normalization before activation functions.
                                        If False, applies it after. Default is True.
        """
        super(IntegratedNN, self).__init__()
        self.batch_norm_before_activation = batch_norm_before_activation
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2 // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2 // 2)
        self.final_dropout = nn.Dropout(p=min(0.2, dropout_prob))
        self.fc4 = nn.Linear(hidden_size // 2 // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            x: torch.Tensor, Input features tensor.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
        if self.batch_norm_before_activation:
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.final_dropout(x)
            x = self.fc4(x)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.bn3(x)
            x = self.final_dropout(x)
            x = self.fc4(x)
        return x
