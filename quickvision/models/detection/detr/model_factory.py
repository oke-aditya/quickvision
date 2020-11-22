import torch
import torch.nn as nn
from quickvision.pretrained._pretrained_weights import _load_pretrained_weights
from quickvision.pretrained._pretrained_detection import detr_weights_dict

__all__ = ["vision_detr", "create_detr_backbone"]


class vision_detr(nn.Module):
    """
    Creates Detr Model for Object Detection.
    Args:
        num_classes: Number of classes to detect.
        num_queries: Number of queries for transformer in Detr.
        backbone: Backbone created from create_detr_backbone.
    """
    def __init__(self, num_classes: int, num_queries: int, backbone: str):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = backbone
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


def create_vision_detr(num_classes: int, num_queries: int, backbone: str):
    """
    Creates Detr Model for Object Detection
    Args:
        num_classes: Number of classes to detect.
        num_queries: Number of queries for transformer in Detr.
        backbone: Backbone created from create_detr_backbone.
    """

    model = vision_detr(num_classes, num_queries, backbone)
    return model


def create_detr_backbone(model_name: str, pretrained: str = None,):
    """
        Creates Detr Backbone for Detection.
        Args:
            model_name: Name of supported bacbone. Supported Backbones are
                  "resnet50", "resnet101", "resnet50_dc5", "resnet101_dc5"
            pretrained: (str) If "coco", returns Detr pretrained on COCO Dataset.
    """
    if(model_name == "resnet50"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)

    elif(model_name == "resnet101"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=False)

    elif(model_name == "resnet50_dc5"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=False)

    elif(model_name == "resnet101_dc5"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=False)

    else:
        raise ValueError("Unuspported backbone")

    if pretrained is not None:
        checkpoint = _load_pretrained_weights(detr_weights_dict, model_name, pretrained=pretrained,)
        backbone.load_state_dict(checkpoint["model"])
        return backbone

    return backbone
