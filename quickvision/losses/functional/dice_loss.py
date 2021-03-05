import torch
from typing import Optional

__all__ = ["dice_loss"]


def dice_loss(
    inputs, targets, weights: Optional[torch.Tensor] = None, reduction: str = "sum"
):
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
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    if weights is not None:
        loss *= weights

    if reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()
