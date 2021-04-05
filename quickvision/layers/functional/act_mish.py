# Taken from Mish author himself https://github.com/digantamisra98/Mish/

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mish"]


@torch.jit.script
def mish(x, inplace: bool = False):
    """
    Applies the mish function element-wise:
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    if inplace:
        return x.mul_(F.tanh(F.softplus(x)))
    else:
        return x.mul(F.tanh(F.softplus(x)))
