import quickvision
import torch
import torchvision
import pytorch_lightning as pl
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

    def test_torchvision(self,):
        print(torchvision.__version__)

    def test_pl(self,):
        print(pl.__version__)


if __name__ == "__main__":
    unittest.main()
