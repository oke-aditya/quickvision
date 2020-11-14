# Port from pytorch_cnn_trainer
# https://github.com/oke-aditya/pytorch_cnn_trainer

import pytorch_lightning as pl
import torch
from quickvision.models.components import create_torchvision_backbone
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy

__all__ = ["lit_cnn"]


class lit_cnn(pl.LightningModule):
    """
    Creates a CNN which can be fine-tuned.
    """
    def __init__(self, pretrained_backbone: str,
                 num_classes: int, learning_rate: float = 0.0001,
                 pretrained: str = None, **kwargs,):
        super().__init__()

        self.num_classes = num_classes
        self.bottom, self.out_channels = create_torchvision_backbone(pretrained_backbone, pretrained)
        self.top = nn.Linear(self.out_channels, self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.bottom(x)
        x = self.top(x.view(-1, self.out_channels))
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        train_loss = F.cross_entropy(outputs, targets, reduction='sum')
        # Possible we can compute top-1 and top-5 accuracy here.
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        val_loss = F.cross_entropy(outputs, targets, reduction='sum')
        # Possible we can compute top-1 and top-5 accuracy here.
        return {"loss": val_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
