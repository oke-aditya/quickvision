import unittest
import torch
import torch.nn as nn
from vision.models.components.torchvision_backbones import create_torchvision_backbone

supported_tv_models = ["resnet18", "vgg13", "resnet50", "mobilenet", "mnasnet0_5", ]
error_model = "invalid_model"


class BackboneTester(unittest.TestCase):
    def test_torchvision_backbones(self):
        for model_name in supported_tv_models:
            ft_backbone, out_channels = create_torchvision_backbone(model_name)
            self.assertTrue(isinstance(ft_backbone, nn.Module))
            self.assertTrue(isinstance(out_channels, int))

    def test_invalid_model(self):
        self.assertRaises(ValueError, create_torchvision_backbone, error_model)


if __name__ == "__main__":
    unittest.main()
