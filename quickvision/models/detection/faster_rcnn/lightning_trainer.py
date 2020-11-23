import pytorch_lightning as pl
import torch
import torch.nn as nn
from quickvision.models.components import create_torchvision_backbone
from quickvision.models.detection.faster_rcnn import create_fastercnn_backbone
from quickvision.models.detection.utils import _evaluate_iou, _evaluate_giou
from torchvision.models.detection.faster_rcnn import (fasterrcnn_resnet50_fpn, FasterRCNN, FastRCNNPredictor,)

__all__ = ["lit_frcnn"]


class lit_frcnn(pl.LightningModule):
    """
    Creates a Faster CNN which can be fine-tuned.
    """

    def __init__(self, learning_rate: float = 0.0001, num_classes: int = 91,
                 backbone: str = None, fpn: bool = True,
                 pretrained_backbone: str = None, trainable_backbone_layers: int = 3, **kwargs,):

        """
        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
            pretrained: if true, returns a model pre-trained on COCO train2017
            pretrained_backbone (str): if "imagenet", returns a model with backbone pre-trained on Imagenet
            trainable_backbone_layers: number of trainable resnet layers starting from final block
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.backbone = backbone
        if backbone is None:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True,
                                                 trainable_backbone_layers=trainable_backbone_layers,)

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        else:
            backbone_model = create_fastercnn_backbone(self.backbone, fpn, pretrained_backbone,
                                                       trainable_backbone_layers, **kwargs,)
            self.model = FasterRCNN(backbone_model, num_classes=num_classes, **kwargs)

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        giou = torch.stack([_evaluate_giou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou, "val_giou": giou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        avg_giou = torch.stack([o["val_giou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou, "val_giou": avg_giou}
        return {"avg_val_iou": avg_iou, "avg_val_giou": avg_giou, "log": logs}

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                               momentum=0.9, weight_decay=0.005,)
