import torch
import torch.nn as nn
import numpy as np
from config.config import CAT_HIDDEN_SIZE, DROPOUT_RATE, REG_HIDDEN_SIZE


class EmbeddingMatrixNN(nn.Module):
    """
    Neural Network model using a pre-trained embedding matrix combined with categorical features for regression.
    This model includes an embedding layer initialized with a given embedding matrix, processes categorical features through
    a fully connected layer, and combines both to predict a continuous target variable.

    Args:
        embedding_matrix: Pre-trained embedding matrix as a NumPy array.
        categorical_dim: Dimension of the categorical feature input.
        cat_hidden_size: Number of neurons in the hidden layer for categorical features.
        reg_hidden_size: Number of neurons in the hidden layer for regression.
        dropout: Dropout probability for regularization.
        requires_grad: If True, the embedding layer weights will be updated during training. Default is False (frozen embeddings).
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        categorical_dim: int,
        cat_hidden_size=CAT_HIDDEN_SIZE,
        reg_hidden_size=REG_HIDDEN_SIZE,
        dropout=DROPOUT_RATE,
        requires_grad=False,
    ):
        super(EmbeddingMatrixNN, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # Frozen/Trainable embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            requires_grad=requires_grad,
        )

        # Layers for categorical features
        self.cat_layer = nn.Linear(categorical_dim, cat_hidden_size)
        self.batch_norm_cat = nn.BatchNorm1d(cat_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        regressor_input_dim = embedding_dim + cat_hidden_size

        # Final regression layers
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, reg_hidden_size),
            nn.BatchNorm1d(reg_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reg_hidden_size, reg_hidden_size // 2),
            nn.BatchNorm1d(reg_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reg_hidden_size // 2, 1),
        )

    def forward(self, text_seq, cat_features):
        # Embed text and average word vectors
        text_emb = self.embedding(text_seq)
        text_mask = (
            (text_seq != 0).unsqueeze(-1).float()
        )  # Create mask for non-padding tokens
        text_feat = (text_emb * text_mask).sum(dim=1) / (
            text_mask.sum(dim=1) + 1e-9
        )  # Average only non-padding vectors

        # Process categorical features
        cat_feat = self.cat_layer(cat_features)
        cat_feat = self.batch_norm_cat(cat_feat)
        cat_feat = self.relu(cat_feat)
        cat_feat = self.dropout(cat_feat)

        # Concatenate all features
        combined = torch.cat((text_feat, cat_feat), dim=1)

        # Final regression layer
        output = self.regressor(combined)
        return output
