"""Multi-Layer Perceptron model."""

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with BatchNorm, ReLU activation, and Dropout.
    """
    
    def __init__(self, input_dim, num_classes, hidden_dims=[1024, 512, 256]):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
        """
        super(MLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)

