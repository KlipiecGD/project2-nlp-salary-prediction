import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.residual_block import ResidualBlock


class SelfTaughtNN(nn.Module):
    """
    Neural Network model combining self-taught text embeddings with categorical features for regression.

    This model includes an embedding layer for text input, processes categorical features through
    a fully connected layer, and combines both to predict a continuous target variable.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        categorical_dim: int,
        cat_hidden_size: int = 128,
        reg_hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        """
        Initializes the SelfTaughtNN model.

        Args:
            vocab_size: int, Size of the vocabulary for the embedding layer.
            embedding_size: int, Dimension of the word embeddings.
            categorical_dim: int, Number of categorical features.
            cat_hidden_size: int, Number of neurons in the hidden layer for categorical features.
            reg_hidden_size: int, Number of neurons in the hidden layer for regression.
            dropout: float, Dropout rate for regularization.
        """
        super(SelfTaughtNN, self).__init__()

        # Trainable embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # Layers for categorical features
        self.cat_layer = nn.Linear(categorical_dim, cat_hidden_size)
        self.batch_norm_cat = nn.BatchNorm1d(cat_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        regressor_input_dim = embedding_size + cat_hidden_size

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

    def forward(self, text_seq, cat_features) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            text_seq: torch.Tensor, Input text sequences (word indices).
            cat_features: torch.Tensor, Categorical features.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
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


class SelfTaughtNNWithResiduals(nn.Module):
    """
    Self-taught NN with Residual Connections.
    This model extends the SelfTaughtNN by incorporating residual blocks in the regression layers.
    As an additional argument, it takes num_residual_blocks to specify how many residual blocks to use (1, 2, or 3).
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
        Initializes the SelfTaughtNNWithResiduals model.
        Args:
            vocab_size: int, Size of the vocabulary for the embedding layer.
            categorical_dim: int, Number of categorical features.
            embedding_size: int, Dimension of the word embeddings, default 300.
            cat_hidden_size: int, Number of neurons in the hidden layer for categorical features, default 128.
            emb_hidden_size: int, Number of neurons in the hidden layers for regression, default 256.
            dropout: float, Dropout rate for regularization, default 0.3.
            num_residual_blocks: int, Number of residual blocks to use in regression (1, 2, or 3), default 2.
        """
        super(SelfTaughtNNWithResiduals, self).__init__()

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


class CNNModel(nn.Module):
    """
    Model using CNN with Global Max Pooling and categorical features.
    This model includes an embedding layer for text input, a convolutional layer followed by
    global max pooling to extract text features, processes categorical features through
    a fully connected layer, and combines both to predict a continuous target variable.
    """

    def __init__(
        self,
        vocab_size: int,
        categorical_dim: int,
        embedding_size: int = 300,
        cat_hidden_size: int = 128,
        emb_hidden_size: int = 256,
        num_filters: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """
        Initializes the CNNModel.
            Args:
                vocab_size: int, Size of the vocabulary for the embedding layer.
                categorical_dim: int, Number of categorical features.
                embedding_size: int, Dimension of the word embeddings, default 300.
                cat_hidden_size: int, Number of neurons in the hidden layer for categorical features, default 128.
                emb_hidden_size: int, Number of neurons in the hidden layers for regression, default 256.
                num_filters: int, Number of filters in the CNN layer, default 64.
                dropout: float, Dropout rate for regularization, default 0.3.
        """
        super(CNNModel, self).__init__()

        # Calculate input dimension for regressor
        # The text features will be of size 'num_filters' after pooling.
        # The categorical features are projected to 'cat_hidden_size'.
        regressor_input_dim = num_filters + cat_hidden_size

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)

        # 2. CNN Layer (Input: [Batch, Embedding_Size, Seq_Length])
        self.conv = nn.Conv1d(
            embedding_size, num_filters, kernel_size=3, padding=1
        )  # 'same' padding
        self.conv_bn = nn.BatchNorm1d(num_filters)

        # 3. Categorical features projection layer
        # Projects cat features to a dimension 'cat_hidden_size' for combination
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 4. Regressor
        self.regressor = nn.Sequential(
            # Input size is num_filters (from CNN) + cat_hidden_size (from Cat Layer)
            nn.Linear(regressor_input_dim, emb_hidden_size),
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
        # 1. Embedding
        text_emb = self.embedding(text_seq)
        text_emb = self.embedding_dropout(text_emb)

        # 2. CNN
        # Transpose from [Batch, Seq_Len, Emb_Dim] to [Batch, Emb_Dim, Seq_Len]
        text_emb_t = text_emb.transpose(1, 2)
        conv_out = F.relu(self.conv(text_emb_t))
        conv_out = self.conv_bn(conv_out)

        # 3. Global Pooling
        # Apply Global Max Pooling across the sequence length (dim 2)
        # This collapses [Batch, Num_Filters, Seq_Length] -> [Batch, Num_Filters, 1]
        # and then squeezes the last dimension to get [Batch, Num_Filters]
        text_feat = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)

        # 4. Categorical features
        cat_feat = self.cat_layer(cat_features)

        # 5. Combine and predict
        combined = torch.cat([text_feat, cat_feat], dim=1)
        output = self.regressor(combined)
        return output


class CNNModelWithResiduals(nn.Module):
    """
    CNN Model with Residual Connections.
    This model extends the CNNModel by incorporating residual blocks in the regression layers.
    It includes an embedding layer for text input, a convolutional layer followed by
    a global max pooling layer, and a fully connected layer for regression.
    """

    def __init__(
        self,
        vocab_size: int,
        categorical_dim: int,
        embedding_size: int = 300,
        cat_hidden_size: int = 128,
        emb_hidden_size: int = 256,
        num_filters: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """
        Initializes the CNNModelWithResiduals.
        Args:
            vocab_size: int, Size of the vocabulary for the embedding layer.
            categorical_dim: int, Number of categorical features.
            embedding_size: int, Dimension of the word embeddings, default 300.
            cat_hidden_size: int, Number of neurons in the hidden layer for categorical features, default 128.
            emb_hidden_size: int, Number of neurons in the hidden layers for regression, default 256.
            num_filters: int, Number of filters in the CNN layer, default 64.
            dropout: float, Dropout rate for regularization, default 0.3.
        """
        super(CNNModelWithResiduals, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)

        # 2. CNN Layers with Residual Connection
        self.conv1 = nn.Conv1d(embedding_size, num_filters, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm1d(num_filters)

        # Additional conv layer for residual connection
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm1d(num_filters)

        # Projection for residual if embedding_size != num_filters
        self.conv_projection = None
        if embedding_size != num_filters:
            self.conv_projection = nn.Conv1d(embedding_size, num_filters, kernel_size=1)

        # 3. Categorical features projection
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 4. Regressor with Residual Blocks
        regressor_input_dim = num_filters + cat_hidden_size

        # First residual block
        self.res_block1 = ResidualBlock(regressor_input_dim, emb_hidden_size, dropout)

        # Second residual block
        self.res_block2 = ResidualBlock(emb_hidden_size, emb_hidden_size // 2, dropout)

        # Final output layer
        self.output_layer = nn.Linear(emb_hidden_size // 2, 1)

    def forward(self, text_seq, cat_features) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            text_seq: torch.Tensor, Input text sequences (word indices).
            cat_features: torch.Tensor, Categorical features.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
        # 1. Embedding
        text_emb = self.embedding(text_seq)
        text_emb = self.embedding_dropout(text_emb)

        # 2. CNN with Residual Connection
        # Transpose: [Batch, Seq_Len, Emb_Dim] -> [Batch, Emb_Dim, Seq_Len]
        text_emb_t = text_emb.transpose(1, 2)

        # First conv
        conv_out = F.relu(self.conv1_bn(self.conv1(text_emb_t)))

        # Second conv with residual
        conv_out2 = self.conv2(conv_out)
        conv_out2 = self.conv2_bn(conv_out2)

        # Residual connection
        conv_out = F.relu(conv_out2 + conv_out)  # Skip connection

        # 3. Global Max Pooling
        text_feat = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)

        # 4. Categorical features
        cat_feat = self.cat_layer(cat_features)

        # 5. Combine
        combined = torch.cat([text_feat, cat_feat], dim=1)

        # 6. Pass through residual blocks
        out = self.res_block1(combined)
        out = self.res_block2(out)

        # 7. Final output
        output = self.output_layer(out)

        return output


class CNNRNNModel(nn.Module):
    """
    Model with CNN and vanilla RNN. Model processes text sequences with a CNN layer followed by a vanilla RNN,
    and combines the extracted features for regression tasks.
    """

    def __init__(
        self,
        vocab_size,
        categorical_dim,
        embedding_size=300,
        cat_hidden_size=128,
        emb_hidden_size=256,
        rnn_hidden_size=128,
        num_filters=64,
        dropout=0.3,
    ) -> None:
        """Initializes the CNNRNNModel.
        Args:
            vocab_size: int, Size of the vocabulary for the embedding layer.
            categorical_dim: int, Number of categorical features.
            embedding_size: int, Dimension of the word embeddings, default 300.
            cat_hidden_size: int, Number of neurons in the hidden layer for categorical features, default 128.
            emb_hidden_size: int, Number of neurons in the hidden layers for regression, default 256.
            rnn_hidden_size: int, Number of neurons in the RNN hidden layer, default 128.
            num_filters: int, Number of filters in the CNN layer, default 64.
            dropout: float, Dropout rate for regularization, default 0.3.
        """
        super(CNNRNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)

        # Single CNN layer
        self.conv = nn.Conv1d(embedding_size, num_filters, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm1d(num_filters)

        # Vanilla RNN
        self.rnn = nn.RNN(
            num_filters,
            rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        # Categorical features
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(rnn_hidden_size + cat_hidden_size, emb_hidden_size),
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
        conv_out = conv_out.transpose(1, 2)

        # RNN
        rnn_out, _ = self.rnn(conv_out)  # _ is final hidden state (not used)

        # Average pooling
        mask = (text_seq != 0).unsqueeze(-1).float()
        text_feat = (rnn_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        # Categorical features
        cat_feat = self.cat_layer(cat_features)

        # Combine and predict
        combined = torch.cat([text_feat, cat_feat], dim=1)
        output = self.regressor(combined)
        return output


class CNNGRUModel(nn.Module):
    """CNN and GRU model with multi-scale CNN and bidirectional GRU.
    This model processes text sequences with multi-scale CNN layers followed by a bidirectional GRU,
    and combines the extracted features with categorical features for regression tasks."""

    def __init__(
        self,
        vocab_size,
        categorical_dim,
        embedding_size=300,
        cat_hidden_size=128,
        emb_hidden_size=256,
        gru_hidden_size=128,
        num_filters=64,
        dropout=0.3,
    ) -> None:
        """Initializes the CNNGRUModel.
        Args:
            vocab_size: int, Size of the vocabulary for the embedding layer.
            categorical_dim: int, Number of categorical features.
            embedding_size: int, Dimension of the word embeddings, default 300.
            cat_hidden_size: int, Number of neurons in the hidden layer for categorical features, default 128.
            emb_hidden_size: int, Number of neurons in the hidden layers for regression, default 256.
            gru_hidden_size: int, Number of neurons in the GRU hidden layer, default 128.
            num_filters: int, Number of filters in the CNN layer, default 64.
            dropout: float, Dropout rate for regularization, default 0.3.
        """
        super(CNNGRUModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)

        # Multi-scale CNN
        self.conv1 = nn.Conv1d(embedding_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_size, num_filters, kernel_size=5, padding=2)
        self.conv_bn = nn.BatchNorm1d(num_filters * 2)

        # Bidirectional GRU
        self.gru = nn.GRU(
            num_filters * 2,
            gru_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Categorical features
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(gru_hidden_size * 2 + cat_hidden_size, emb_hidden_size),
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
        text_emb = self.embedding(text_seq)
        text_emb = self.embedding_dropout(text_emb)

        # Multi-scale CNN
        text_emb_t = text_emb.transpose(1, 2)
        conv1_out = F.relu(self.conv1(text_emb_t))
        conv2_out = F.relu(self.conv2(text_emb_t))
        conv_out = torch.cat([conv1_out, conv2_out], dim=1)
        conv_out = self.conv_bn(conv_out)
        conv_out = conv_out.transpose(1, 2)

        # Bidirectional GRU
        gru_out, _ = self.gru(conv_out)

        # Max and average pooling
        mask = (text_seq != 0).unsqueeze(-1).float()
        gru_out_masked = gru_out * mask
        max_pool = torch.max(gru_out_masked, dim=1)[0]
        avg_pool = gru_out_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        text_feat = torch.cat([max_pool, avg_pool], dim=1)

        cat_feat = self.cat_layer(cat_features)
        combined = torch.cat([text_feat, cat_feat], dim=1)
        output = self.regressor(combined)
        return output


class CNNLSTMModel(nn.Module):
    """Model with CNN and LSTM.
    This model processes text sequences with multi-scale CNN layers followed by a bidirectional LSTM,
    and combines the extracted features with categorical features for regression tasks."""

    def __init__(
        self,
        vocab_size,
        categorical_dim,
        embedding_size=300,
        cat_hidden_size=128,
        emb_hidden_size=256,
        lstm_hidden_size=128,
        num_filters=64,
        dropout=0.3,
    ) -> None:
        """Initializes the CNNLSTMModel.
        Args:
            vocab_size: int, Size of the vocabulary for the embedding layer.
            categorical_dim: int, Number of categorical features.
            embedding_size: int, Dimension of the word embeddings, default 300.
            cat_hidden_size: int, Number of neurons in the hidden layer for categorical features, default 128.
            emb_hidden_size: int, Number of neurons in the hidden layers for regression, default 256.
            lstm_hidden_size: int, Number of neurons in the LSTM hidden layer, default 128.
            num_filters: int, Number of filters in the CNN layer, default 64.
            dropout: float, Dropout rate for regularization, default 0.3.
        """
        super(CNNLSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)

        # Multi-scale CNN
        self.conv1 = nn.Conv1d(embedding_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_size, num_filters, kernel_size=5, padding=2)
        self.conv_bn = nn.BatchNorm1d(num_filters * 2)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            num_filters * 2,
            lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Categorical features
        self.cat_layer = nn.Sequential(
            nn.Linear(categorical_dim, cat_hidden_size),
            nn.BatchNorm1d(cat_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2 + cat_hidden_size, emb_hidden_size),
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
            torch.Tensor, Model output.
        """
        text_emb = self.embedding(text_seq)
        text_emb = self.embedding_dropout(text_emb)

        # Multi-scale CNN
        text_emb_t = text_emb.transpose(1, 2)
        conv1_out = F.relu(self.conv1(text_emb_t))
        conv2_out = F.relu(self.conv2(text_emb_t))
        conv_out = torch.cat([conv1_out, conv2_out], dim=1)
        conv_out = self.conv_bn(conv_out)
        conv_out = conv_out.transpose(1, 2)

        # Bidirectional LSTM
        lstm_out, _ = self.lstm(conv_out)

        # Max and average pooling
        mask = (text_seq != 0).unsqueeze(-1).float()
        lstm_out_masked = lstm_out * mask
        max_pool = torch.max(lstm_out_masked, dim=1)[0]
        avg_pool = lstm_out_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        text_feat = torch.cat([max_pool, avg_pool], dim=1)

        cat_feat = self.cat_layer(cat_features)
        combined = torch.cat([text_feat, cat_feat], dim=1)
        output = self.regressor(combined)
        return output
