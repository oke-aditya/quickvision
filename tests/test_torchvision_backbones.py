import unittest
import torch
import torch.nn as nn
from quickvision.models.components.torchvision_backbones import create_torchvision_backbone

supported_tv_models = ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18",
                       "resnet34", "resnet50", "resnet101", "resnet152",
                       "resnext50_32x4d", "resnext101_32x8d",
                       "mnasnet0_5", "mnasnet1_0", "mobilenet_v2",
                       "wide_resnet50_2", "wide_resnet101_2"]

error_model = "invalid_model"


class BackboneTester(unittest.TestCase):
    def test_torchvision_backbones(self):
        for model_name in supported_tv_models:
            ft_backbone, out_channels = create_torchvision_backbone(model_name, pretrained=None)
            self.assertTrue(isinstance(ft_backbone, nn.Module))
            self.assertTrue(isinstance(out_channels, int))

    def test_invalid_model(self):
        self.assertRaises(ValueError, create_torchvision_backbone, error_model)

    def test_torchvision_imagenet_backbones(self):
        for model_name in supported_tv_models:
            ft_backbone, out_channels = create_torchvision_backbone(model_name, pretrained="imagenet")
            self.assertTrue(isinstance(ft_backbone, nn.Module))
            self.assertTrue(isinstance(out_channels, int))


if __name__ == "__main__":
    unittest.main()
