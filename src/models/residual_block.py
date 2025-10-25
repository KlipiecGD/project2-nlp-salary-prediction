import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block: output = F(x) + x
    If dimensions don't match, uses a projection layer.
    This block consists of two linear layers with Batch Normalization and ReLU activation,
    along with dropout for regularization.
    """

    def __init__(self, input_dim, output_dim, dropout=0.3) -> None:
        """Initializes the ResidualBlock.
        Args:
            input_dim: int, Dimension of the input features.
            output_dim: int, Dimension of the output features.
            dropout: float, Dropout rate for regularization. Default is 0.3.
        """
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection layer if dimensions don't match
        self.projection = None
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        """Defines the forward pass of the ResidualBlock.
        Args:
            x: torch.Tensor, Input features tensor.
        """
        identity = x

        # First layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Second layer
        out = self.fc2(out)
        out = self.bn2(out)

        # Projection if needed
        if self.projection is not None:
            identity = self.projection(identity)

        # Residual connection
        out = out + identity
        out = F.relu(out)

        return out
