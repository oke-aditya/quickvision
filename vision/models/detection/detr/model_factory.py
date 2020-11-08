import torch
import torch.nn as nn

__all__ = ["vision_detr", "create_detr_backbone"]


class vision_detr(nn.Module):
    """
    Creates Detr Model for Object Detection.
    Args:
        num_classes: Number of classes to detect.
        num_queries: Number of queries for transformer in Detr.
        backbone: Backbone created from create_detr_backbone.
    """
    def __init__(self, num_classes, num_queries, backbone):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = backbone
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


def create_vision_detr(num_classes, num_queries, backbone):
    """
    Creates Detr Model for Object Detection
    Args:
        num_classes: Number of classes to detect.
        num_queries: Number of queries for transformer in Detr.
        backbone: Backbone created from create_detr_backbone.
    """

    model = vision_detr(num_classes, num_queries, backbone)
    return model


def create_detr_backbone(name: str, pretrained: bool = True,):
    """
        Creates Detr Backbone for Detection.
        Args:
            name: Name of supported bacbone. Supported Backbones are
                  "resnet50", "resnet101", "resnet50_dc5", "resnet101_dc5"
            pretrained: If True, returns backbone pretrained on COCO Dataset.
    """
    if(name == "resnet50"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=pretrained)

    elif(name == "resnet101"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=pretrained)

    elif(name == "resnet50_dc5"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=pretrained)

    elif(name == "resnet101_dc5"):
        backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50_dc5', pretrained=pretrained)

    else:
        raise ValueError("Unuspported backbone")

    return backbone
