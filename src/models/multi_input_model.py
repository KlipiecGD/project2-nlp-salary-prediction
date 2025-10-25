import torch
import torch.nn as nn


class MultiInputNN(nn.Module):
    """
    A neural network designed to handle multiple input types, such as text embeddings and tabular data.

    The network consists of three main components: a sub-network for processing embeddings,
    a sub-network for processing tabular features, and a combined network that
    concatenates their outputs to produce a final prediction. Each sub-network
    uses a sequence of linear layers, batch normalization, ReLU activation, and dropout.
    """

    def __init__(
        self,
        embedding_dim: int,
        tabular_dim: int,
        embedding_hidden: list[int] = [256, 128],
        tabular_hidden: list[int] = [64, 32],
        combined_hidden: list[int] = [128, 64],
        dropout_prob: float = 0.3,
    ) -> None:
        """
        Initializes the MultiInputNN model.

        Args:
            embedding_dim: int, The dimension of the input text embeddings.
            tabular_dim: int, The number of features in the tabular data.
            embedding_hidden: list[int], A list of integers representing the number of neurons in
                              the hidden layers of the embedding sub-network. Defaults to [256, 128].
            tabular_hidden: list[int], A list of integers representing the number of neurons in
                            the hidden layers of the tabular sub-network. Defaults to [64, 32].
            combined_hidden: list[int], A list of integers representing the number of neurons in
                             the hidden layers of the combined network. Defaults to [128, 64].
            dropout_prob: float, The dropout probability applied in all dropout layers. Defaults to 0.3.
        """
        super(MultiInputNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.tabular_dim = tabular_dim
        self.embedding_hidden = embedding_hidden
        self.tabular_hidden = tabular_hidden
        self.dropout_prob = dropout_prob

        # Embedding layers
        embedding_layers = []
        input_size = embedding_dim
        for hidden_size in embedding_hidden:
            embedding_layers.append(nn.Linear(input_size, hidden_size))
            embedding_layers.append(nn.BatchNorm1d(hidden_size))
            embedding_layers.append(nn.ReLU())
            embedding_layers.append(nn.Dropout(p=dropout_prob))
            input_size = hidden_size

        self.embedding_net = nn.Sequential(*embedding_layers)

        # Tabular layers
        tabular_layers = []
        input_size = tabular_dim
        for hidden_size in tabular_hidden:
            tabular_layers.append(nn.Linear(input_size, hidden_size))
            tabular_layers.append(nn.BatchNorm1d(hidden_size))
            tabular_layers.append(nn.ReLU())
            tabular_layers.append(nn.Dropout(p=dropout_prob))
            input_size = hidden_size

        self.tabular_net = nn.Sequential(*tabular_layers)

        # Combined layers
        combined_input_size = embedding_hidden[-1] + tabular_hidden[-1]
        combined_layers = []
        for hidden_size in combined_hidden:
            combined_layers.append(nn.Linear(combined_input_size, hidden_size))
            combined_layers.append(nn.BatchNorm1d(hidden_size))
            combined_layers.append(nn.ReLU())
            combined_layers.append(nn.Dropout(p=dropout_prob))
            combined_input_size = hidden_size

        combined_layers.append(nn.Linear(combined_input_size, 1))  # Final output layer
        self.combined_net = nn.Sequential(*combined_layers)

    def forward(self, embeddings: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            embeddings: torch.Tensor, Input text embeddings.
            tabular: torch.Tensor, Input tabular features.
        Returns:
            torch.Tensor: Predicted continuous target value.
        """
        embeddings = self.embedding_net(embeddings)
        tabular = self.tabular_net(tabular)
        combined = torch.cat((embeddings, tabular), dim=1)
        output = self.combined_net(combined)
        return output
