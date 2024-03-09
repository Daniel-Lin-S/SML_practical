"""Implementations of MLP, ResNet, and Transformer models for tabular data."""

import enum
import math
import torch
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class MLP(nn.Module):
    """
    Implements a multi-layer perceptron (MLP) for classification, with improvements.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[32, 64, 128],
        dropout=0.0,
        activation="tanh",
    ):
        """
        Initializes the MLP model with improved practices.
        Args:
            - input_dim (int): Dimensionality of input features.
            - output_dim (int): Dimensionality of output classes.
            - hidden_dims (list): List of integers representing the sizes of hidden layers.
            - dropout (float): Dropout rate.
            - activation (str): Activation function to use ("tanh" or "relu")
        """
        super(MLP, self).__init__()
        assert len(hidden_dims) > 0, "hidden_dims should have at least one element"

        # Create the fully connected layers based on hidden_dims
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Activation function {activation} not supported")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Final layer without activation before the loss
        self.fc_layers = nn.Sequential(*layers)
        self.fc_final = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the MLP.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor before softmax.
        """

        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.fc_final(x)
        return x


class ResNet(nn.Module):
    pass


# Those implementations are incomplete and are only used for demonstration purposes
##### DO NOT USE IN PRODUCTION #####

# the code below is from the authors [gorishniy2021revisiting]
# we only use numerical feature tokenizer in this project


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_TokenInitialization":
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x, d) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class NumericalFeatureTokenizer(nn.Module):
    """Transforms continuous features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    For one feature, the transformation consists of two steps:

    * the feature is multiplied by a trainable vector
    * another trainable vector is added

    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x):
        """Embeddings have the same shape as the input tensor."""
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class AttEncoder(nn.Module):
    """
    A simple implementation of a Transformer encoder for tabular data.

    The model consists of the following components:
        - Embedding layer for numerical features
        - Multi-head self-attention layer
        - Final linear layer for classification
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        n_tokens: int,
        linear_hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ):
        """
        Initializes the AttEncoder model.

        Args:
            - input_dim (int): Dimensionality of input features.
            - embed_dim (int): Dimensionality of the embeddings.
            - n_tokens (int): Number of tokens to use.
            - linear_hidden_dim (int): Dimensionality of the hidden layer.
            - output_dim (int): Dimensionality of the output classes.
            - num_heads (int): Number of heads in the multi-head attention layer.
            - dropout_rate (float): Dropout rate.
        """

        self.input_net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
        )

        self.feature_tokenizer = NumericalFeatureTokenizer(
            input_dim, n_tokens, bias=True, initialization="uniform"
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.output_net = nn.Sequential(
            nn.Linear(embed_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_hidden_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        tokens = self.feature_tokenizer(x)
        x = self.input_net(x)

    pass





class GroupedFeaturesTransformer(nn.Module):
    def __init__(
        self,
        n_groups,
        n_features_per_group,
        n_hidden,
        n_classes,
        n_heads,
        n_encoder_layers,
        dropout=0.1,
    ):
        """
        Model insipired by Vit (Dosovitskiy et al., 2021) for tabular data.

        Args:
            - n_groups (int): Number of feature groups.
            - n_features_per_group (list): Number of features in each group.
            - n_hidden (int): Dimensionality of the hidden layer.
            - n_classes (int): Dimensionality of the output classes.
            - n_heads (int): Number of heads in the multi-head attention layer.
            - n_encoder_layers (int): Number of layers in the Transformer encoder.
            - dropout (float): Dropout rate.
        """

        super(GroupedFeaturesTransformer, self).__init__()
        self.n_groups = n_groups
        self.n_hidden = n_hidden

        self.embeddings = nn.ModuleList(
            [nn.Linear(n_features, n_hidden) for n_features in n_features_per_group]
        )

        self.n_feat_group = [0] + np.cumsum(n_features_per_group).tolist()
        # print(self.n_feat_group)

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(n_hidden) for _ in range(n_groups)]
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=128,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(n_hidden)

        # Replacing the single group classification with a global pooling operation
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x_embedded = []

        for i, (embedding, norm) in enumerate(zip(self.embeddings, self.layer_norms)):
            group = x[:, self.n_feat_group[i] : self.n_feat_group[i + 1]]
            # print(group.shape)

            embedded_group = embedding(group.reshape(batch_size, -1))

            normed_group = norm(embedded_group)

            x_embedded.append(normed_group)

        x = torch.stack(x_embedded, dim=1)

        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        x = self.transformer_encoder(x)
        x = self.encoder_norm(x)

        # Apply global average pooling across all groups
        x = x.permute(1, 2, 0)  # [batch_size, embedding_dim, seq_len]
        
        x = self.global_pooling(x).squeeze(-1)  # [batch_size, embedding_dim]

        # Classifier
        x = self.classifier(x)
        return x
