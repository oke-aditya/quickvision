import unittest
import torch
from typing import Dict
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from torch_utils import im2tensor
from vision.models.detection import detr
from vision.models import model_utils
from vision.models.detection.detr import engine
from dataset_utils import DummyDetectionDataset

if(torch.cuda.is_available()):
    from torch.cuda import amp

train_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_classes=2, num_samples=10)
val_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_classes=2, num_samples=10)

supported_detr_backbones = ["resnet50", "resnet50_dc5", ]
#  "resnet101", "resnet101_dc5"]


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                           shuffle=False, collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                         shuffle=False, collate_fn=collate_fn)


class ModelFactoryTester(unittest.TestCase):
    def test_detr_backbones(self):
        for supp_bb in supported_detr_backbones:
            bbone = detr.create_detr_backbone(supp_bb, pretrained=False)
            self.assertTrue(isinstance(bbone, nn.Module))

    def test_vision_detr(self):
        for supp_bb in supported_detr_backbones:
            bbone = detr.create_detr_backbone(supp_bb, pretrained=False)
            self.assertTrue(isinstance(bbone, nn.Module))
            model = detr.vision_detr(num_classes=91, num_queries=5, backbone=bbone)
            self.assertTrue(isinstance(bbone, nn.Module))

    def test_create_vision_detr(self):
        for supp_bb in supported_detr_backbones:
            bbone = detr.create_detr_backbone(supp_bb, pretrained=False)
            self.assertTrue(isinstance(bbone, nn.Module))
            model = detr.create_vision_detr(num_classes=91, num_queries=5, backbone=bbone)
            self.assertTrue(isinstance(bbone, nn.Module))


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
        return True

    def test_infer(self):
        # Infer over an image
        image = Image.open("tests/assets/grace_hopper_517x606.jpg")
        tensor = im2tensor(image)
        self.assertEqual(tensor.ndim, 4)
        return True

    def test_train_step(self):
        pass

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_step_cuda(self):
        pass

    def test_val_step(self):
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


class LightningTester(unittest.TestCase):
    def test_lit_detr(self):
        flag = False
        for bbone in supported_detr_backbones:
            model = detr.lit_detr(num_classes=2, num_queries=5, pretrained=False, backbone=bbone)
            trainer = pl.Trainer(fast_dev_run=True)
            trainer.fit(model, train_loader, val_loader)
        flag = True
        self.assertTrue(flag)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_lit_detr_cuda(self):
        flag = False
        for bbone in supported_detr_backbones:
            model = detr.lit_detr(num_classes=2, num_queries=5, pretrained=False, backbone=bbone)
            trainer = pl.Trainer(fast_dev_run=True)
            trainer.fit(model, train_loader, val_loader)
        flag = True
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
