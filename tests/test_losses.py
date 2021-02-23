import torch
import torch.nn as nn
import unittest
from quickvision import losses
from quickvision.losses import functional as fl


class DiceLossTester(unittest.TestCase):
    def test_dice_loss_functional(self):
        # Same thing what you do with below.
        # inputs = torch.tensor([[0.4, 0.2, 0.3], [0.6, 0.2, 0.3]], dtype=torch.float32)
        # targets = torch.tensor([[0], [1]], dtype=torch.float32)
        # loss = fl.dice_loss(inputs, targets)
        # Do a backward
        # loss.backward()
        # And now compare this loss with known valueQ
        # self.assertTrue()
        pass
0
    def test_dice_loss(self):
        # loss_fn = losses.DiceLoss()
        # inputs = torch.tensor([[0.4, 0.2, 0.3], [0.6, 0.2, 0.3]], dtype=torch.float32)
        # targets = torch.tensor([[0], [1]], dtype=torch.float32)
        # loss = loss_fn(inputs, targets)
        # See what expected loss should be
        # expected_loss = []
        # Compare those two with epsilon
        # loss.backward()
        # Assert those with epsilon case
        # self.assertTrue()
        pass


if __name__ == "__main__":
    unittest.main()
