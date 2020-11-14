import vision
import torch
import torchvision
import vision.pretrained
import vision.layers
import vision.models
import vision.optimizers
import vision.utils
import vision.tensorrt


def test_torch():
    print(torch.__version__)
    return 1
