import torch
import torch.nn as nn


class LinearNet(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        """Initializes a linear neural network with multiple layers.

        This takes the format expected by Pyannote and builds a simple feedforward
        neural network with the specified number of layers and hidden units.

        Args:
            input_dim (int): The number of input features.
            hidden_size (int): The number of features in the hidden layers.
            num_layers (int): The number of hidden layers in the network.
        """
        super().__init__()
        layers = []
        self.in_features = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_features = hidden_size if num_layers > 0 else input_dim

        if num_layers <= 0:  # No hidden layers, identity mapping
            self.layers = nn.Identity()
            return

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
