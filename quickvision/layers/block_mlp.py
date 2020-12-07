import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MLP"]


class MLP(nn.Module):
    """
    A very simple Multi Layered Perceptron classifier.
    Args:
        in_features (int): Input features of Network
        hidden_features (int): Intermediate Features of network
        out_features (int): Output layers of network.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.l1 = nn.Linear(self.in_features, self.hidden_features)
        self.l2 = nn.Linear(self.hidden_features, self.out_features)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return (x)
