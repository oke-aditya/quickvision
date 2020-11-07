import torch
import torch.nn as nn

__all__ = ["detr_model"]


class detr_model(nn.Module):
    def __init__(self, num_classes, num_queries, backbone, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load("facebookresearch/detr", backbone, pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(
            in_features=self.in_features, out_features=self.num_classes
        )
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)
