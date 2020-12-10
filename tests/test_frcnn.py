import unittest
import torch
from typing import Dict
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from torch_utils import im2tensor
from quickvision.models.detection import faster_rcnn
from quickvision.models.detection.faster_rcnn import engine
from dataset_utils import DummyDetectionDataset

if(torch.cuda.is_available()):
    from torch.cuda import amp

fpn_supported_models = ["resnet18", ]  # "resnet34","resnet50", "resnet101", "resnet152",
#  "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"

non_fpn_supported_models = ["mobilenet_v2"]
# "resnet18", "resnet34", "resnet50","resnet101",
# "resnet152", "resnext101_32x8d", "mobilenet_v2", "vgg11", "vgg13", "vgg16", "vgg19"

train_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_classes=2, class_start=1,
                                      num_samples=10, box_fmt="xyxy")
val_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_classes=2, class_start=1,
                                    num_samples=10, box_fmt="xyxy")


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                           shuffle=False, collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2,
                                         shuffle=False, collate_fn=collate_fn)


class ModelFactoryTester(unittest.TestCase):
    def test_frcnn_fpn(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))

            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))

    def test_frcnn_nonfpn(self):
        for bbone in non_fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None, fpn=False)
            self.assertTrue(isinstance(backbone, nn.Module))

            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))


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
        frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=5)
        out = frcnn_model(img_tensor, targets)
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
        frcnn_model = faster_rcnn.create_vision_fastercnn()
        frcnn_model.eval()
        out = frcnn_model(tensor)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)

    def test_train_step_fpn(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            opt = torch.optim.SGD(frcnn_model.parameters(), lr=1e-3)
            train_metrics = faster_rcnn.train_step(frcnn_model, train_loader, "cpu", opt, num_batches=10)
            self.assertIsInstance(train_metrics, Dict)
            exp_keys = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in train_metrics.keys())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_step_fpn_cuda(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            opt = torch.optim.SGD(frcnn_model.parameters(), lr=1e-3)
            train_metrics = faster_rcnn.train_step(frcnn_model, train_loader, "cuda", opt, num_batches=10)
            self.assertIsInstance(train_metrics, Dict)
            exp_keys = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in train_metrics.keys())

    def test_val_step_fpn(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            val_metrics = faster_rcnn.val_step(frcnn_model, train_loader, "cpu", num_batches=10)
            self.assertIsInstance(val_metrics, Dict)
            exp_keys = ("iou", "giou")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in val_metrics.keys())

    def test_fit(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            opt = torch.optim.SGD(frcnn_model.parameters(), lr=1e-3)
            history = faster_rcnn.fit(frcnn_model, 1, train_loader, val_loader, "cpu", opt, num_batches=10)

            self.assertIsInstance(history, Dict)
            exp_keys = ("train", "val")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in history.keys())
            # exp_keys2 = ("train_loss")
            # for exp_k2 in exp_keys2:
            #     self.assertTrue(exp_k2 in history["train"].keys())
            # exp_keys3 = ("val_iou", "val_giou")
            # for exp_k3 in exp_keys3:
            #     self.assertTrue(exp_k3 in history["val"].keys())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_fit_cuda(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            opt = torch.optim.SGD(frcnn_model.parameters(), lr=1e-3)
            history = faster_rcnn.fit(frcnn_model, 1, train_loader, val_loader, "cuda", opt, num_batches=4, fp16=True)

            self.assertIsInstance(history, Dict)
            exp_keys = ("train", "val")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in history.keys())
            # exp_keys2 = ("train_loss")
            # for exp_k2 in exp_keys2:
            #     self.assertTrue(exp_k2 in history["train"].keys())
            # exp_keys3 = ("val_iou", "val_giou")
            # for exp_k3 in exp_keys3:
            #     self.assertTrue(exp_k3 in history["val"].keys())

    def test_train_sanity_fit(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            result = faster_rcnn.train_sanity_fit(frcnn_model, train_loader, "cpu", num_batches=10)
            self.assertTrue(result)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_sanity_fit_cuda(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            result = faster_rcnn.train_sanity_fit(frcnn_model, train_loader, "cuda", num_batches=10, fp16=True)
            self.assertTrue(result)

    def test_val_sanity_fit(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            result = faster_rcnn.val_sanity_fit(frcnn_model, val_loader, "cpu", num_batches=10)
            self.assertTrue(result)

    def test_sanity_fit(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            result = faster_rcnn.sanity_fit(frcnn_model, train_loader, val_loader, "cpu", num_batches=10)
            self.assertTrue(result)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_sanity_fit_cuda(self):
        for bbone in fpn_supported_models:
            backbone = faster_rcnn.create_fastercnn_backbone(backbone=bbone, pretrained=None)
            self.assertTrue(isinstance(backbone, nn.Module))
            frcnn_model = faster_rcnn.create_vision_fastercnn(num_classes=3, backbone=backbone)
            self.assertTrue(isinstance(frcnn_model, nn.Module))
            result = faster_rcnn.sanity_fit(frcnn_model, train_loader, val_loader, "cuda", num_batches=10, fp16=True)
            self.assertTrue(result)


class LightningTester(unittest.TestCase):
    def test_lit_frcnn_fpn(self):
        flag = False
        for bbone in fpn_supported_models:
            model = faster_rcnn.lit_frcnn(num_classes=3, backbone=bbone, fpn=True, pretrained_backbone=False,)
            trainer = pl.Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
            trainer.fit(model, train_loader, val_loader)
        flag = True
        self.assertTrue(flag)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_lit_cnn_cuda(self):
        flag = False
        for bbone in fpn_supported_models:
            model = faster_rcnn.lit_frcnn(num_classes=3, backbone=bbone, fpn=True, pretrained_backbone=False,)
            trainer = pl.Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
            trainer.fit(model, train_loader, val_loader)
        flag = True
        self.assertTrue(flag)

    def test_lit_forward(self):
        model = faster_rcnn.lit_frcnn(num_classes=3, pretrained_backbone=False)
        image = torch.rand(1, 3, 400, 400)
        out = model(image)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
