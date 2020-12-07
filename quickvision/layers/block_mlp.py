import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MLP"]


class MLP(nn.Module):
    """
    A very simple Multi Layered Perecptron classifier
    """

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l1 = nn.Linear(self.in_features, self.out_channels)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        return (x)
