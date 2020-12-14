import unittest
import torch
import torch.nn as nn
from quickvision.models.classification import cnn
from PIL import Image
from dataset_utils import create_cifar10_dataset, create_loaders
from torch_utils import im2tensor
from typing import Dict
import pytorch_lightning as pl
from torch.optim.swa_utils import SWALR

if(torch.cuda.is_available()):
    from torch.cuda import amp

supported_tv_models = ["resnet18",
                       # "resnet34", # "resnet50", # "resnet101", # "resnet152",
                       # "resnext50_32x4d", # "resnext101_32x8d", # "vgg11",
                       # "vgg13", # "vgg16", # "vgg19", # "mobilenet", # "mnasnet0_5",
                       # "mnasnet1_0",]
                       ]

supported_timm_models = ["resnet18"]

# @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")

train_ds, val_ds = create_cifar10_dataset()
train_loader, val_loader = create_loaders(train_ds, val_ds, num_workers=1)


class ModelFactoryTester(unittest.TestCase):
    def test_cnn(self):
        for model_name in supported_tv_models:
            model = cnn.cnn(model_name, 10, pretrained=None)
            self.assertTrue(isinstance(model, nn.Module))

    def test_create_cnn(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            self.assertTrue(isinstance(model, nn.Module))


class cnnTester(unittest.TestCase):
    def test_train(self):
        # Read Image using PIL Here
        # Do forward over image
        image = Image.open("tests/assets/grace_hopper_517x606.jpg")
        tensor = im2tensor(image)
        self.assertEqual(tensor.ndim, 4)
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            out = model(tensor)
            self.assertEqual(out.shape[1], 10)
            self.assertEqual(out.ndim, 2)

    def test_infer(self):
        # Infer over an image
        image = Image.open("tests/assets/grace_hopper_517x606.jpg")
        tensor = im2tensor(image)
        self.assertEqual(tensor.ndim, 4)
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            model = model.eval()
            out = model(tensor)
            self.assertEqual(out.shape[1], 10)
            self.assertEqual(out.ndim, 2)

    def test_train_step(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-4, max_lr=1e-3, mode="min")
            train_metrics = cnn.train_step(model, train_loader, loss, "cpu", opt, num_batches=10,
                                           grad_penalty=True)
            self.assertIsInstance(train_metrics, Dict)
            exp_keys = ("loss", "top1", "top5")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in train_metrics.keys())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_step_cuda(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-4, max_lr=1e-3, mode="min")
            scaler = amp.GradScaler()
            train_metrics = cnn.train_step(model, train_loader, loss, "cuda", opt,
                                           num_batches=10, scaler=scaler)
            self.assertIsInstance(train_metrics, Dict)
            exp_keys = ("loss", "top1", "top5")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in train_metrics.keys())

    def test_val_step(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            loss = nn.CrossEntropyLoss()
            val_metrics = cnn.val_step(model, val_loader, loss, "cpu", num_batches=10)
            self.assertIsInstance(val_metrics, Dict)
            exp_keys = ("loss", "top1", "top5")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in val_metrics.keys())

    def test_fit(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-4, max_lr=1e-3, mode="min")
            history = cnn.fit(model, 1, train_loader, val_loader, loss, device="cpu",
                              optimizer=opt, num_batches=10)
            self.assertIsInstance(history, Dict)
            exp_keys = ("train", "val")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in history.keys())

            exp_keys2 = ("top1_acc", "top5_acc", "loss")
            for exp_k2 in exp_keys2:
                self.assertTrue(exp_k2 in history["train"].keys())
                self.assertTrue(exp_k2 in history["val"].keys())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_fit_cuda(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-4, max_lr=1e-3, mode="min")
            history = cnn.fit(model, 1, train_loader, val_loader, loss, device="cuda",
                              optimizer=opt, num_batches=10, fp16=True)

            self.assertIsInstance(history, Dict)
            exp_keys = ("train", "val")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in history.keys())

            exp_keys2 = ("top1_acc", "top5_acc", "loss")
            for exp_k2 in exp_keys2:
                self.assertTrue(exp_k2 in history["train"].keys())
                self.assertTrue(exp_k2 in history["val"].keys())

    def test_fit_swa(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
            swa_scheduler = SWALR(opt, anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
            swa_start = 2
            history = cnn.fit(model, 4, train_loader, val_loader, loss, device="cpu",
                              optimizer=opt, scheduler=scheduler, num_batches=10,
                              swa_start=swa_start, swa_scheduler=swa_scheduler)
            self.assertIsInstance(history, Dict)
            exp_keys = ("train", "val")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in history.keys())

            exp_keys2 = ("top1_acc", "top5_acc", "loss")
            for exp_k2 in exp_keys2:
                self.assertTrue(exp_k2 in history["train"].keys())
                self.assertTrue(exp_k2 in history["val"].keys())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_fit_swa_cuda(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
            swa_scheduler = SWALR(opt, anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
            swa_start = 2
            history = cnn.fit(model, 3, train_loader, val_loader, loss, device="cpu",
                              optimizer=opt, scheduler=scheduler, num_batches=10,
                              swa_start=swa_start, swa_scheduler=swa_scheduler)
            self.assertIsInstance(history, Dict)
            exp_keys = ("train", "val")
            for exp_k in exp_keys:
                self.assertTrue(exp_k in history.keys())

            exp_keys2 = ("top1_acc", "top5_acc", "loss")
            for exp_k2 in exp_keys2:
                self.assertTrue(exp_k2 in history["train"].keys())
                self.assertTrue(exp_k2 in history["val"].keys())

    def test_train_sanity_fit(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            res = cnn.train_sanity_fit(model, train_loader, loss, "cpu", num_batches=10)
            self.assertTrue(res)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_train_sanity_fit_cuda(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()
            res = cnn.train_sanity_fit(model, train_loader, loss, "cuda", num_batches=10, fp16=True)
            self.assertTrue(res)

    def test_val_sanity_fit(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            loss = nn.CrossEntropyLoss()
            res = cnn.val_sanity_fit(model, val_loader, loss, "cpu", num_batches=10)
            self.assertTrue(res)

    def test_sanity_fit(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            model = model.cpu()
            loss = nn.CrossEntropyLoss()
            res = cnn.sanity_fit(model, train_loader, val_loader, loss, "cpu", num_batches=10)
            self.assertTrue(res)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_sanity_fit_cuda(self):
        for model_name in supported_tv_models:
            model = cnn.create_cnn(model_name, 10, pretrained=None)
            loss = nn.CrossEntropyLoss()
            res = cnn.sanity_fit(model, train_loader, val_loader, loss, "cuda", num_batches=10, fp16=True)
            self.assertTrue(res)


class LightningTester(unittest.TestCase):
    def test_lit_cnn(self):
        flag = False
        for model_name in supported_tv_models:
            model = cnn.LitCNN(model_name, num_classes=10, pretrained=None)
            trainer = pl.Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
            trainer.fit(model, train_loader, val_loader)
        flag = True
        self.assertTrue(flag)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_lit_cnn_cuda(self):
        flag = False
        for model_name in supported_tv_models:
            model = cnn.LitCNN(model_name, num_classes=10, pretrained=None)
            trainer = pl.Trainer(fast_dev_run=True, logger=False, checkpoint_callback=False)
            trainer.fit(model, train_loader, val_loader)
        flag = True
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
