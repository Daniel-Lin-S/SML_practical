import torch
import numpy as np
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
