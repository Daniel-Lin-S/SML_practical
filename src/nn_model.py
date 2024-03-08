import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Implements a multi-layer perceptron (MLP) for classification, with improvements.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[32, 64, 128], dropout=0.0):
        """
        Initializes the MLP model with improved practices.
        Args:
            input_dim (int): Dimensionality of input features.
            output_dim (int): Dimensionality of output classes.
            hidden_dims (list): List of integers representing the sizes of hidden layers.
            dropout (float): Dropout rate.
        """
        super(MLP, self).__init__()
        assert len(hidden_dims) > 0, "hidden_dims should have at least one element"

        # Create the fully connected layers based on hidden_dims
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Final layer without a ReLU activation before the loss
        self.fc_layers = nn.Sequential(*layers)
        self.fc_final = nn.Linear(hidden_dims[-1], output_dim)

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        """
        Defines the forward pass of the MLP.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor before softmax.
        """
        x = self.fc_layers(x)
        x = self.fc_final(x)
        return x

    def _init_weights(self):
        """
        Initialize weights using Kaiming (He) initialization for better training performance.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
