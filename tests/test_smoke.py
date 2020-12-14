import quickvision
import torch
import torchvision
import quickvision.pretrained
import quickvision.layers
import quickvision.models
import quickvision.optimizers
import quickvision.utils
import unittest


class HelloTester(unittest.TestCase):
    def test_torch(self,):
        print(torch.__version__)
        return True
