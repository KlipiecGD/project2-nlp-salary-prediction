import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.residual_block import ResidualBlock


class UnfrozenResidualNN(nn.Module):
    """
    Residual Neural Network with unfrozen embeddings and categorical features.
    Includes a flexible number of residual blocks for regression tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        categorical_dim: int,
        embedding_size: int = 300,
        cat_hidden_size: int = 128,
        emb_hidden_size: int = 256,
        dropout: float = 0.3,
        num_residual_blocks: int = 2,
    ) -> None:
        """
        Initializes the UnfrozenResidualNN model.
        Args:
            vocab_size: Size of the vocabulary for the embedding layer.
            categorical_dim: Number of categorical features.
            embedding_size: Dimension of the word embeddings. Defaults to 300.
            cat_hidden_size: Number of neurons in the hidden layer for categorical features. Defaults to 128.
            emb_hidden_size: Number of neurons in the hidden layers for regression. Defaults to 256.
            dropout: Dropout rate for regularization. Defaults to 0.3.
            num_residual_blocks: Number of residual blocks to use in regression (1, 2, or 3). Defaults to 2.
        """
        super(UnfrozenResidualNN, self).__init__()

        # Trainable embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # Layers for categorical features
        self.cat_layer = nn.Linear(categorical_dim, cat_hidden_size)
        self.batch_norm_cat = nn.BatchNorm1d(cat_hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Combined dimension
        regressor_input_dim = embedding_size + cat_hidden_size

        # Build regressor with flexible number of residual blocks
        self.num_blocks = num_residual_blocks

        if num_residual_blocks == 1:
            # Simpler: Just one block
            self.res_block1 = ResidualBlock(
                regressor_input_dim, emb_hidden_size, dropout
            )
            self.output_layer = nn.Linear(emb_hidden_size, 1)

        elif num_residual_blocks == 2:
            # Standard: Two blocks (RECOMMENDED - matches original depth)
            self.res_block1 = ResidualBlock(
                regressor_input_dim, emb_hidden_size, dropout
            )
            self.res_block2 = ResidualBlock(
                emb_hidden_size, emb_hidden_size // 2, dropout
            )
            self.output_layer = nn.Linear(emb_hidden_size // 2, 1)

        elif num_residual_blocks == 3:
            # Deeper: Three blocks (for larger datasets)
            self.res_block1 = ResidualBlock(
                regressor_input_dim, emb_hidden_size, dropout
            )
            self.res_block2 = ResidualBlock(
                emb_hidden_size, emb_hidden_size // 2, dropout
            )
            self.res_block3 = ResidualBlock(
                emb_hidden_size // 2, emb_hidden_size // 2, dropout
            )
            self.output_layer = nn.Linear(emb_hidden_size // 2, 1)

        else:
            raise ValueError("num_residual_blocks must be 1, 2, or 3")

    def forward(self, text_seq, cat_features) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            text_seq: torch.Tensor, Input text sequences (word indices).
            cat_features: torch.Tensor, Categorical features.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
        # 1. Embed text and average word vectors
        text_emb = self.embedding(text_seq)
        text_mask = (text_seq != 0).unsqueeze(-1).float()
        text_feat = (text_emb * text_mask).sum(dim=1) / (text_mask.sum(dim=1) + 1e-9)

        # 2. Process categorical features
        cat_feat = self.cat_layer(cat_features)
        cat_feat = self.batch_norm_cat(cat_feat)
        cat_feat = F.relu(cat_feat)
        cat_feat = self.dropout(cat_feat)

        # 3. Concatenate features
        combined = torch.cat((text_feat, cat_feat), dim=1)

        # 4. Pass through residual blocks
        if self.num_blocks == 1:
            out = self.res_block1(combined)
        elif self.num_blocks == 2:
            out = self.res_block1(combined)
            out = self.res_block2(out)
        elif self.num_blocks == 3:
            out = self.res_block1(combined)
            out = self.res_block2(out)
            out = self.res_block3(out)

        # 5. Final output
        output = self.output_layer(out)

        return output


class UnfrozenCNN(nn.Module):
    """CNN model with unfrozen pre-trained embeddings and categorical features.
    Combines CNN-extracted text features with categorical features for regression tasks.
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        categorical_dim: int,
        cat_hidden_size: int = 128,
        emb_hidden_size: int = 256,
        num_filters: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """Initializes the UnfrozenCNN model.
        Args:
            embedding_matrix: Pre-trained embedding matrix as a NumPy array.
            categorical_dim: Dimension of the categorical feature input.
            cat_hidden_size: Number of neurons in the hidden layer for categorical features. Defaults to 128.
            emb_hidden_size: Number of neurons in the hidden layers for regression. Defaults to 256.
            num_filters: Number of filters in the CNN layer. Defaults to 64.
            dropout: Dropout probability for regularization. Defaults to 0.3.
        """
        super(UnfrozenCNN, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # Unfrozen embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=True
        )
        self.embedding_dropout = nn.Dropout(0.2)

        # CNN Layer
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm1d(num_filters)

        # Categorical features projection layer
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(num_filters + cat_hidden_size, emb_hidden_size),
            nn.BatchNorm1d(emb_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_hidden_size, emb_hidden_size // 2),
            nn.BatchNorm1d(emb_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_hidden_size // 2, 1),
        )

    def forward(self, text_seq, cat_features) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            text_seq: torch.Tensor, Input text sequences (word indices).
            cat_features: torch.Tensor, Categorical features.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
        # Embedding
        text_emb = self.embedding(text_seq)
        text_emb = self.embedding_dropout(text_emb)

        # CNN
        text_emb_t = text_emb.transpose(1, 2)
        conv_out = F.relu(self.conv(text_emb_t))
        conv_out = self.conv_bn(conv_out)

        # Global Max Pooling
        text_feat = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)

        # Categorical features
        cat_feat = self.cat_layer(cat_features)

        # Combine and predict
        combined = torch.cat([text_feat, cat_feat], dim=1)
        output = self.regressor(combined)
        return output


class UnfrozenCNNWithResiduals(nn.Module):
    """Unfrozen CNN model with Residual Connections and categorical features.
    Combines CNN-extracted text features with categorical features for regression tasks,
    and includes residual blocks in the regressor for enhanced learning.
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        categorical_dim: int,
        cat_hidden_size: int = 128,
        emb_hidden_size: int = 256,
        num_filters: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """Initializes the UnfrozenCNNWithResiduals model.
        Args:
            embedding_matrix: Pre-trained embedding matrix as a NumPy array.
            categorical_dim: Dimension of the categorical feature input.
            cat_hidden_size: Number of neurons in the hidden layer for categorical features. Defaults to 128.
            emb_hidden_size: Number of neurons in the hidden layers for regression. Defaults to 256.
            num_filters: Number of filters in the CNN layer. Defaults to 64.
            dropout: Dropout probability for regularization. Defaults to 0.3.
        """
        super(UnfrozenCNNWithResiduals, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # Unfrozen embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=True
        )
        self.embedding_dropout = nn.Dropout(0.2)

        # CNN Layers with Residual Connection
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm1d(num_filters)

        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm1d(num_filters)

        # Categorical features projection
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regressor with Residual Blocks
        regressor_input_dim = num_filters + cat_hidden_size

        self.res_block1 = ResidualBlock(regressor_input_dim, emb_hidden_size, dropout)
        self.res_block2 = ResidualBlock(emb_hidden_size, emb_hidden_size // 2, dropout)

        self.output_layer = nn.Linear(emb_hidden_size // 2, 1)

    def forward(self, text_seq, cat_features) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            text_seq: torch.Tensor, Input text sequences (word indices).
            cat_features: torch.Tensor, Categorical features.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
        # Embedding
        text_emb = self.embedding(text_seq)
        text_emb = self.embedding_dropout(text_emb)

        # CNN with Residual Connection
        text_emb_t = text_emb.transpose(1, 2)

        conv_out = F.relu(self.conv1_bn(self.conv1(text_emb_t)))

        conv_out2 = self.conv2(conv_out)
        conv_out2 = self.conv2_bn(conv_out2)

        conv_out = F.relu(conv_out2 + conv_out)  # Skip connection

        # Global Max Pooling
        text_feat = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)

        # Categorical Features
        cat_feat = self.cat_layer(cat_features)

        # Concatenate Features
        combined_feat = torch.cat((text_feat, cat_feat), dim=1)

        # Residual Blocks
        x = self.res_block1(combined_feat)
        x = self.res_block2(x)

        # Output Layer
        output = self.output_layer(x)
        return output
