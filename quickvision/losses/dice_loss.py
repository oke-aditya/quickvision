import torch
import torch.nn as nn
from quickvision.losses.functional import dice_loss
from typing import Optional

__all__ = ["DiceLoss"]


class DiceLoss(nn.Module):
    """
    Computes the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        weights: A float tensor of weights for each sample.
        reduction: A string specifying whether loss should be a sum or average.
    """

    def __init__(self, weights: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()

        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        return dice_loss(inputs, targets, self.weights, self.reduction)
