import torch
import torch.nn as nn
from config.config import DROPOUT_RATE


class IntegratedNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        dropout_prob: float = DROPOUT_RATE,
        batch_norm_before_activation: bool = True,
    ):
        """
        An integrated feed-forward neural network with configurable batch normalization placement.

        This model is a multi-layered perceptron designed for regression, featuring
        batch normalization and dropout for improved training stability and generalization.
        The order of batch normalization and activation can be specified.

        Args:
            input_dim: The number of input features.
            hidden_size: The number of neurons in the first hidden layer. Subsequent
                         layers will have a decreasing number of neurons. Defaults to 256.
            dropout_prob: The dropout probability applied after each activation.
            batch_norm_before_activation: If True, batch normalization is applied before
                                          the ReLU activation. If False, it is applied after.
                                          Defaults to True.
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
