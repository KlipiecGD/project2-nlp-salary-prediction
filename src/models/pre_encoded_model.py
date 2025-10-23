import torch
import torch.nn as nn
from config.config import CAT_HIDDEN_SIZE, DROPOUT_RATE, REG_HIDDEN_SIZE


class PreEncodedModel(nn.Module):
    """
    Model for pre-encoded text embeddings combined with categorical features.
    It processes categorical features through a hidden layer, combines them with
    the pre-encoded text vectors, and predicts a continuous target variable.

    Args:
        text_vector_dim: Dimension of the pre-encoded text vectors.
        categorical_dim: Dimension of the categorical feature input.
        cat_hidden_size: Number of neurons in the hidden layer for categorical features.
        reg_hidden_size: Number of neurons in the hidden layer for regression.
        dropout: Dropout probability for regularization.
    """

    def __init__(
        self,
        text_vector_dim: int,
        categorical_dim: int,
        cat_hidden_size: int = CAT_HIDDEN_SIZE,
        reg_hidden_size: int = REG_HIDDEN_SIZE,
        dropout: float = DROPOUT_RATE,
    ):
        super(PreEncodedModel, self).__init__()

        # Process categorical features
        self.cat_layer = nn.Linear(categorical_dim, cat_hidden_size)
        self.batch_norm_cat = nn.BatchNorm1d(cat_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Combine text vectors and categorical features
        combined_dim = text_vector_dim + cat_hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, reg_hidden_size),
            nn.BatchNorm1d(reg_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reg_hidden_size, reg_hidden_size // 2),
            nn.BatchNorm1d(reg_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reg_hidden_size // 2, 1),
        )

    def forward(self, text_vectors, cat_features):
        # No embedding needed

        # Process categorical features
        cat_feat = self.cat_layer(cat_features)
        cat_feat = self.batch_norm_cat(cat_feat)
        cat_feat = self.relu(cat_feat)
        cat_feat = self.dropout(cat_feat)

        # Concatenate pre-encoded text with categorical features
        combined = torch.cat((text_vectors, cat_feat), dim=1)

        return self.regressor(combined)
