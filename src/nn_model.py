"""Implementations of MLP, ResNet, and Transformer models for tabular data."""

import enum
import math
import torch
import skorch
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

# wrap for sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from src.dataloader import MusicData
from src.data_utils import MusicDataset
from src.train_nn import *
import torch


class PatchEmbedding(nn.Module):
    """
    Projects the groups of features into a sequence of patches.

    The input is K groups of features (may be distinct sizes),

    for each group, there is a corresponding linear projection to a low-dimensional embedding space.

    The resulting embeddings are then concatenated and projected to the desired embedding size.

    e.g. Split 500 features into 10 groups of 50, and project each group to 4 dimensions.
    then the resulting sequence will be length 10, and each element will be of size 4.
    """

    def __init__(self, input_dim, n_groups, n_features_per_group, n_embed_dim):
        """
        Initializes the PatchEmbedding layer.

        Args:
            - input_dim (int): The dimensionality of the input tensor
            - n_groups (int): The number of feature groups
            - n_features_per_group (list): The number of features in each group
            - n_embed_dim (int): The dimensionality of the embedding
        """
        super().__init__()

        self.input_dim = input_dim

        self.n_groups = n_groups

        self.n_embed_dim = n_embed_dim

        self.n_feat_group = [0] + np.cumsum(n_features_per_group).tolist()

        # Create linear layers for each group
        self.projections = nn.ModuleList(
            [nn.Linear(n_features, n_embed_dim) for n_features in n_features_per_group]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(n_embed_dim) for _ in range(n_groups)]
        )

    def forward(self, x):
        """
        Args:
            - x (torch.Tensor): The input tensor of shape [batch_size, n_features]

        Returns:
            - x (torch.Tensor): The output tensor of shape [batch_size, n_patches, n_embed_dim]
        """
        x_embedded = []

        for i, (projection, norm) in enumerate(zip(self.projections, self.layer_norms)):
            group = x[:, self.n_feat_group[i] : self.n_feat_group[i + 1]]
            embedded_group = projection(group)
            normed_group = norm(embedded_group)
            x_embedded.append(normed_group)

        x = torch.cat(x_embedded, dim=1).reshape(x.size(0), -1, self.n_embed_dim)

        return x

    @property
    def output_dim(self):
        return self.n_embed_dim

    @property
    def token_seq_length(self):
        return self.n_groups


class MLP(nn.Module):
    """
    Implements a multi-layer perceptron (MLP) for classification, with improvements.
    """

    def __init__(
        self,
        input_dim,
        output_dim=8,
        n_groups=11,
        n_features_per_group=[84, 84, 84, 140, 7, 7, 7, 49, 7, 42, 7],
        n_embed_dim=8,
        hidden_dims=[32, 64, 128],
        dropout=0.0,
        activation="relu",
        use_patch_embedding=False,
    ):
        """
        Initializes the MLP model.
        Args:
            - input_dim (int): Dimensionality of input features.
            - output_dim (int): Dimensionality of output classes.
            - n_groups (int): Number of feature groups.
            - n_features_per_group (list): Number of features in each group.
            - n_embed_dim (int): Dimensionality of the embedding.
            - hidden_dims (list): List of integers representing the sizes of hidden layers.
            - dropout (float): Dropout rate.
            - activation (str): Activation function to use ("tanh" or "relu" or "sigmoid").
            - use_patch_embedding (bool): Whether to use the PatchEmbedding layer.
        """
        super().__init__()
        
        assert len(hidden_dims) > 0, "hidden_dims should have at least one element"

        if use_patch_embedding:
            # if uses patch embedding, then start from n_groups * n_embed_dim
            self.patch_embedding = PatchEmbedding(
                input_dim=input_dim,
                n_groups=n_groups,
                n_features_per_group=n_features_per_group,
                n_embed_dim=n_embed_dim,
            )
            self.extra_fc = nn.Linear(n_groups * n_embed_dim, hidden_dims[0])
            prev_dim = n_groups * n_embed_dim

        else:
            self.patch_embedding = None
            prev_dim = input_dim

        # Create the fully connected layers based on hidden_dims
        layers = []

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Activation function {activation} not supported")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Create sequential model
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

        if self.patch_embedding is not None:
            x = self.patch_embedding(x)
            x = x.reshape(x.size(0), -1)

        x = self.fc_layers(x)
        x = self.fc_final(x)
        return x


class ResNetBlock(nn.Module):
    """
    Initializes a single block of the ResNet model."

    ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))
    """

    def __init__(self, dim, dropout=0.0, normalization="layer_norm"):
        """
        Initializes the ResNetBlock.

        Args:
            - dim (int): The dimensionality of the input tensor
            - dropout (float): Dropout rate
            - normalization (str): The normalization to use ("layer_norm" or "batch_norm")
        """
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(dim)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"Normalization {normalization} not supported")

    def forward(self, x):
        """
        Defines the forward pass of the ResNetBlock.
        """

        # TODO: distinguish between batch and layer norm
        identity = x
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += identity
        return x


class ResNet(nn.Module):
    """
    Implements a simple ResNet model for classification.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        final_layer_dim,
        n_groups,
        n_features_per_group,
        n_embed_dim,
        block_size=32,
        num_blocks=3,
        dropout=0.5,
        normalization="layer_norm",
        use_patch_embedding=True,
    ):
        """
        Initializes the ResNet model.

        Args:
            - input_dim (int): The dimensionality of the input tensor
            - output_dim (int): The dimensionality of the output classes
            - final_layer_dim (int): The dimensionality of the final layer
            - n_groups (int): The number of feature groups
            - n_features_per_group (list): The number of features in each group
            - n_embed_dim (int): The dimensionality of the embedding
            - block_sizes (int): The size of the ResNet blocks
            - num_blocks (int): The number of ResNet blocks
            - dropout (float): Dropout rate
            - normalization (str): The normalization to use ("layer_norm" or "batch_norm")
            - use_patch_embedding (bool): Whether to use the PatchEmbedding layer
        """

        super().__init__()
        
        self.dropout = dropout
        
        if use_patch_embedding:
            self.patch_embedding = PatchEmbedding(
                input_dim=input_dim,
                n_groups=n_groups,
                n_features_per_group=n_features_per_group,
                n_embed_dim=n_embed_dim,
            )
            # this requires flattening of input
            self.down_sample = nn.Linear(n_groups * n_embed_dim, block_size)
            print(f"Using patch embedding, the input dim is {n_groups * n_embed_dim}")
        else:
            self.patch_embedding = None
            self.down_sample = nn.Linear(input_dim, block_size)
            

        # Create the ResNet blocks
        self.resnet_blocks = nn.ModuleList(
            [
                ResNetBlock(
                    block_size, dropout=dropout, normalization=normalization
                )
                for _ in range(num_blocks)
            ]
        )

        # prediction layer
        self.penultimate_layer = nn.Linear(block_size, final_layer_dim)
        self.output_layer = nn.Linear(final_layer_dim, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.
        """
        if self.patch_embedding is not None:
            x = self.patch_embedding(x)
            x = x.reshape(x.size(0), -1)
            
        x = self.down_sample(x)

        for block in self.resnet_blocks:
            x = block(x)

        x = self.penultimate_layer(x)
        x = F.relu(x)
        # x = nn.Dropout(self.dropout)(x)
        
        x = self.output_layer(x)

        return x


def scaled_dot_product(q, k, v, mask=None):
    """
    Computes the scaled dot-product attention.

    Args:
        - q (torch.Tensor): The query tensor
        - k (torch.Tensor): The key tensor
        - v (torch.Tensor): The value tensor
        - mask (torch.Tensor): The mask tensor
    """
    d_k = q.size()[-1]

    attn_logits = torch.matmul(q, k.transpose(-2, -1))

    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = F.softmax(attn_logits, dim=-1)

    values = torch.matmul(attention, v)

    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        """
        Implements the multi-head attention layer.

        This first projects the already learned embeddings into stacked [Q, K, V] matrices,
        and then applies the scaled dot-product attention.

        Finally, the output is projected back to the original embedding dimension.

        Args:
            - input_dim (int): The dimensionality of the input tensor
            - embed_dim (int): The dimensionality of the embedding
            - num_heads (int): The number of heads

        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension {embed_dim} must be 0 modulo number of heads {num_heads}."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """Takes input [Batch, SeqLen, Dims] and returns [Batch, SeqLen, Dims]."""
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Implements a single encoder block of the Transformer model.

        Args:
            - input_dim (int): The dimensionality of the input tensor
            - num_heads (int): The number of heads
            - dim_feedforward (int): The dimensionality of the feedforward layer
            - dropout (float): Dropout rate
        """

        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Outputs the same dimensions as the input."""
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerPredictor(nn.Module):
    """
    Implements the transformer with possible embeddings using custom functions.
    (to visualise attention)
    """

    def __init__(
        self,
        input_dim,
        n_groups,
        n_heads,
        n_features_per_group,
        n_embed_dim,
        output_dim,
        num_layers,
        dim_feedforward,
        dropout=0.0,
        use_patch_embedding=True,
        token_size=7,
        use_average_pooling=False,
    ):
        """
        Initializes the Transformer model.
        If not using patch embedding, the input tensor is divided into
        sequence of tokens of size token_size.

        Args:
            - input_dim (int): The dimensionality of the input tensor
            - n_groups (int): The number of feature groups
            - n_heads (int): The number of heads in the multi-head attention layer
            - n_features_per_group (list): The number of features in each group
            - n_embed_dim (int): The dimensionality of the embedding
            - output_dim (int): The dimensionality of the output classes
            - num_layers (int): The number of layers in the model
            - dim_feedforward (int): The dimensionality of the feedforward layer
            - dropout (float): Dropout rate
            - use_patch_embedding (bool): Whether to use the PatchEmbedding layer
            - token_size (int): The size of the token
            - use_average_pooling (bool): Whether to use average pooling, if not
                then linear layer is used instead
        """
        super().__init__()

        if use_average_pooling:
            raise NotImplementedError("Not implemented without average pooling")

        self.num_layers = num_layers

        if use_patch_embedding:
            self.patch_embedding = PatchEmbedding(
                input_dim=input_dim,
                n_groups=n_groups,
                n_features_per_group=n_features_per_group,
                n_embed_dim=n_embed_dim,
            )
            model_dim = n_embed_dim
            # Output layer (can be implemented as a global pooling operation)
            # but here simply a linear layer that takes (batch, seq, features) -> (batch, classes)
            self.output_layer = nn.Linear(
                in_features=model_dim * n_groups, out_features=output_dim
            )
        

        else:
            self.patch_embedding = None
            assert (
                int(input_dim) % token_size == 0
            ), f"Input dimension {input_dim} must be divisible by token size {token_size}"
            
            self.token_size = token_size
            
            model_dim = token_size
            
            n_heads = 1
            
            self.output_layer = nn.Linear(
                in_features=input_dim, out_features=output_dim
            )
            
            # raise NotImplementedError("Not implemented without patch embedding")

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    input_dim=model_dim,
                    num_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        """
        Defines the forward pass of the Transformer model.

        Args:
            - x (torch.Tensor): The input tensor, shape [batch_size, n_features]
            - mask (torch.Tensor): The mask tensor

        Returns:
            - x (torch.Tensor): The output tensor, shape [batch_size, n_classes]
        """
        if self.patch_embedding is not None:
            x = self.patch_embedding(x)

        else:
            # divide (batch, input) -> (batch, seq_len, token_size)
            x = x.view(x.size(0), -1, self.token_size)

        for layer in self.encoder:
            x = layer(x, mask=mask)

        # when using only linear layer, need to resize the tensor
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        return x

    @torch.no_grad()
    def get_attention_map(self, x, mask=None):
        """
        Returns the attention map for the input tensor.
        """
        if self.patch_embedding is not None:
            x = self.patch_embedding(x)

        attention_maps = []
        for layer in self.encoder:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)

        return attention_maps


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
    
    

class ConvVAE(nn.Module):
    """Implementation of Convolutional VAE model."""
    def __init__(self, 
                 latent_dim = 8, 
                dropout = 0.0,
                 hidden_dims = [64, 32]):
        """
        Initializes the VAE model.   
        
        Args:  
            - input_dim (int): Dimensionality of input features.
            - latent_dim (int): Dimensionality of the latent space.
            - dropout (float): Dropout rate.
            - hidden_dims (list): List of integers representing the sizes of hidden layers.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(64*7*7,latent_dim)
        self.fc_var = nn.Linear(64*7*7,latent_dim) 
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64*7*7),
            nn.Unflatten(1,(64,7,7)),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,3,padding=1,output_padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,padding=1,output_padding=1,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Tanh()
        )
        
        self.latent = torch.empty(latent_dim)
        self.mu = None
        self.log_var = None
        
    def _reparametrize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        Returns:  
            - x (torch.Tensor): The decoded output tensor.
            - z (torch.Tensor): The latent space tensor.
            - mu (torch.Tensor): The mean of the latent space.
            - logvar (torch.Tensor): The log-variance of the latent space.
        """
        latents = self.encoder(x)
        
        mu = self.fc_mu(latents)
        
        logvar = self.fc_var(latents)
        
        
        z = self._reparametrize(mu, logvar)
        
        x = self.decoder(z)
        
        # store the latent space
        self.mu = mu
        self.logvar = logvar
        return x
    
    def sample_latents(self, x):
        """Sample from the latent space."""
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=-1)
        z = self._reparametrize(mu, logvar)
        return z
    
    

class BetaVAE(nn.Module):
    """Implementation of BetaVAE model."""
    def __init__(self, 
                 input_dim, 
                 latent_dim = 8, 
                #  activation = "relu",
                dropout = 0.0,
                 hidden_dims = [64, 32]):
        """
        Initializes the VAE model.   
        
        Args:  
            - input_dim (int): Dimensionality of input features.
            - latent_dim (int): Dimensionality of the latent space.
            - dropout (float): Dropout rate.
            - hidden_dims (list): List of integers representing the sizes of hidden layers.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        self.before_latent_dim = latent_dim * 2
        
        self.encoder_hidden_dims = [input_dim] + hidden_dims
        self.decoder_hidden_dims = [latent_dim] + hidden_dims[::-1]
        
        encoder_layers = []
        decoder_layers = []
        
        # Encoder
        for i in range(len(self.encoder_hidden_dims) - 1):
            encoder_layers.append(nn.Linear(self.encoder_hidden_dims[i], self.encoder_hidden_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            if self.dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Encoder to mu and logvar
        self.fc_mu = nn.Linear(hidden_dims[-1],latent_dim)
        
        # require nonnegative logvar
        self.fc_var = nn.Linear(hidden_dims[-1],latent_dim)
        
        # Decoder
        for i in range(len(self.decoder_hidden_dims) - 1):
            # before the output
            decoder_layers.append(nn.Linear(self.decoder_hidden_dims[i], self.decoder_hidden_dims[i + 1]))
            decoder_layers.append(nn.ReLU())
            if self.dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
        # output layer
        decoder_layers.append(nn.Linear(self.decoder_hidden_dims[-1], 
                                        self.input_dim))
                
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.mu = None
        self.logvar = None
        
    def _reparametrize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        Returns:  
            - x (torch.Tensor): The decoded output tensor.
            - z (torch.Tensor): The latent space tensor.
            - mu (torch.Tensor): The mean of the latent space.
            - logvar (torch.Tensor): The log-variance of the latent space.
        """
        # check and reshape

        all_latents = self.encoder(x)
        
        mu = self.fc_mu(all_latents)
        logvar = self.fc_var(all_latents)
        
        z = self._reparametrize(mu, logvar)
        x = self.decoder(z)
        
        # store the latent space
        self.mu = mu
        self.logvar = logvar
        return x
    
    def extract_mean(self, x):
        """Extract the mean of the latent space."""
        all_latents = self.encoder(x)
        mu = self.fc_mu(all_latents)
        return mu
    
    def extract_logvar(self, x):
        """Extract the logvar of the latent space."""
        all_latents = self.encoder(x)
        logvar = self.fc_var(all_latents)
        return logvar


class SklearnWrappedMLP(BaseEstimator, ClassifierMixin):
    """
    Wrapper for MLP model to use in sklearn pipeline.
    """

    def __init__(
        self,
        input_dim,
        output_dim=8,
        n_groups=11,
        n_features_per_group=[84, 84, 84, 140, 7, 7, 7, 49, 7, 42, 7],
        n_embed_dim=8,
        hidden_dims=[32, 64, 128],
        dropout=0.0,
        activation="relu",
        use_patch_embedding=False,
        batch_size=64,
        epochs = 100,
        patience = 20,
        device = "cpu"
    ):
        """
        Initializes the MLP model.
        Args:
            - input_dim (int): Dimensionality of input features.
            - output_dim (int): Dimensionality of output classes.
            - n_groups (int): Number of feature groups.
            - n_features_per_group (list): Number of features in each group.
            - n_embed_dim (int): Dimensionality of the embedding.
            - hidden_dims (list): List of integers representing the sizes of hidden layers.
            - dropout (float): Dropout rate.
            - activation (str): Activation function to use ("tanh" or "relu" or "sigmoid").
            - use_patch_embedding (bool): Whether to use the PatchEmbedding layer.
            - batch_size (int): The batch size for training.
            - epochs (int): Number of epochs to train the model.
            - patience (int): Number of epochs to wait before early stopping.
            - device (str): The device to use for training.
        """
        
        # configure the model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_groups = n_groups
        self.n_features_per_group = n_features_per_group
        self.n_embed_dim = n_embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.use_patch_embedding = use_patch_embedding
        
        # configure the training
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device

        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            n_groups=n_groups,
            n_features_per_group=n_features_per_group,
            n_embed_dim=n_embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_patch_embedding=use_patch_embedding,
        )

    def fit(self, X, y):
        """
        Fits the model to the given data.
        Args:
            - X (np.ndarray): The input data.
            - y (np.ndarray): The labels.
        Returns:
            - self
        """
        # convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # train the model
        train_loader = DataLoader(
            MusicData(X, y), batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.model.to(self.device)
        
        for epoch in range(self.epochs):
            train_loss, train_acc = train(
                model=self.model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_function=nn.CrossEntropyLoss(),
                scheduler=None,
                device=self.device,
            )
        
        return self
    
    def predict(self, X):
        """
        Predicts the labels for the given data.
        Args:
            - X (np.ndarray): The input data.
        Returns:
            - y_pred (np.ndarray): The predicted labels.
        """
        # convert to torch tensor
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # predict
        with torch.no_grad():
            self.model.eval()
            try:
                y_pred = self.model(X).argmax(dim=1).numpy()
            except:
                y_pred = self.model(X).cpu().argmax(dim=1).numpy()

        return y_pred
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    