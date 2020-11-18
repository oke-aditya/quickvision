import pytorch_lightning as pl
import torch
import torch.nn as nn
from quickvision.losses import detr_loss
from quickvision.models.detection.detr import create_detr_backbone

__all__ = ["lit_detr"]


class lit_detr(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, num_classes: int = 91,
                 num_queries: int = 5, pretrained: str = None,
                 backbone: str = "resnet50", **kwargs, ):

        """
            PyTorch Lightning implementation of `Detr: End-to-End Object Detection with Transformers
            During training, the model expects both the input tensors, as well as targets (list of dictionary),
            containing:
                - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
                - labels (`Int64Tensor[N]`): the class label for each ground truh box
            Args:
                learning_rate: the learning rate
                num_classes: number of detection classes (including background)
                num_queries: number of queries to the transformer module.
                pretrained: if "coco", returns a model pre-trained on COCO train2017
                backbone:  Supported Detection backbones are "resnet50", "resnet101", "resnet50_dc5", "resnet101_dc5".

            It returns a dict with the following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                        Shape= [batch_size x num_queries x (num_classes + 1)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                        (center_x, center_y, height, width). These values are normalized in [0, 1],
                        relative to the size of each individual image (disregarding possible padding).
                        See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        super().__init__()

        self.model = create_detr_backbone(backbone, pretrained)
        in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features=in_features, out_features=num_classes)
        self.model.num_queries = num_queries
        self.learning_rate = learning_rate

        matcher = detr_loss.HungarianMatcher()
        weight_dict = {"loss_ce": 1, "loss_bbox": 1, "loss_giou": 1}
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = detr_loss.SetCriterion(num_classes - 1, matcher, weight_dict, eos_coef=0.5, losses=losses)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)

        self.criterion.train()
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)

        self.criterion.eval()
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return {"loss": loss, "log": loss_dict}

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                               momentum=0.9, weight_decay=0.005,)
