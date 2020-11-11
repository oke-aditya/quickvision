import unittest
import torch
from typing import Dict
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from torch_utils import im2tensor
from vision.models import model_utils
from vision.models.detection import retinanet
from dataset_utils import DummyDetectionDataset

if(torch.cuda.is_available()):
    from torch.cuda import amp

fpn_supported_models = ["resnet18", ]  # "resnet34","resnet50", "resnet101", "resnet152",
#  "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"

non_fpn_supported_models = ["mobilenet_v2"]
# "resnet18", "resnet34", "resnet50","resnet101",
# "resnet152", "resnext101_32x8d", "mobilenet_v2", "vgg11", "vgg13", "vgg16", "vgg19"

train_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_classes=2,
                                      num_samples=10, box_fmt="xyxy")
val_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_classes=2,
                                    num_samples=10, box_fmt="xyxy")


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                           shuffle=False, collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                         shuffle=False, collate_fn=collate_fn)


class ModelFactoryTester(unittest.TestCase):
    def test_retina_fpn(self):
        for bbone in fpn_supported_models:
            backbone = retinanet.create_retinanet_backbone(name=bbone, pretrained=False)
            self.assertTrue(isinstance(backbone, nn.Module))

            retina_model = retinanet.create_vision_retinanet(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(retina_model, nn.Module))

    def test_retina_nonfpn(self):
        for bbone in non_fpn_supported_models:
            backbone = retinanet.create_retinanet_backbone(name=bbone, pretrained=False, fpn=False)
            self.assertTrue(isinstance(backbone, nn.Module))

            retina_model = retinanet.create_vision_retinanet(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(retina_model, nn.Module))


class EngineTester(unittest.TestCase):
    def test_train(self):
        # Read Image using PIL Here
        # Do forward over image
        image = Image.open("tests/assets/grace_hopper_517x606.jpg")
        img_tensor = im2tensor(image)
        self.assertEqual(img_tensor.ndim, 4)
        boxes = torch.tensor([[0, 0, 100, 100], [0, 1, 2, 2],
                             [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)
        labels = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        targets = [{"boxes": boxes, "labels": labels}]
        retina_model = retinanet.create_vision_fastercnn(num_classes=5)
        out = retina_model(img_tensor, targets)
        self.assertIsInstance(out, Dict)
        self.assertIsInstance(out["loss_classifier"], torch.Tensor)
        self.assertIsInstance(out["loss_box_reg"], torch.Tensor)
        self.assertIsInstance(out["loss_objectness"], torch.Tensor)
        self.assertIsInstance(out["loss_rpn_box_reg"], torch.Tensor)

    def test_infer(self):
        # Infer over an image
        image = Image.open("tests/assets/grace_hopper_517x606.jpg")
        tensor = im2tensor(image)
        self.assertEqual(tensor.ndim, 4)
        retina_model = retinanet.create_vision_fastercnn()
        retina_model.eval()
        out = retina_model(tensor)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)

    def test_train_step_fpn(self):
        pass

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_step_fpn_cuda(self):
        pass

    def test_val_step_fpn(self):
        pass

    def test_fit(self):
        pass

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_fit_cuda(self):
        pass

    def test_train_sanity_fit(self):
        pass

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_sanity_fit_cuda(self):
        pass

    def test_val_sanity_fit(self):
        pass

    def test_sanity_fit(self):
        pass

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_sanity_fit_cuda(self):
        pass
