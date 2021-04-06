
import torch.nn as nn
from quickvision.pretrained._model_zoo import TORCHVISION_MODEL_ZOO
from quickvision.pretrained._pretrained_weights import _load_pretrained_weights
from quickvision.pretrained._pretrained_cnns import WEIGHTS_DICT

__all__ = ["create_torchvision_backbone"]

RESNET_SMALL_MODELS = ["resnet18", "resnet34"]
RESNET_LARGE_MODELS = ["resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
                       "wide_resnet50_2", "wide_resnet101_2"]
VGG_MODELS = ["vgg11", "vgg13", "vgg16", "vgg19", ]
MNASNET_MODELS = ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]
MOBILENET_MODELS = ["mobilenet_v2", "mobilenet_v3_large"]


def _create_backbone_generic(model: nn.Module, out_channels: int):
    """
    Generic Backbone creater. It removes the last linear layer.
    Args:
        model: torch.nn model
        out_channels: Number of out_channels in last layer.
    """
    modules_total = list(model.children())
    modules = modules_total[:-1]
    ft_backbone = nn.Sequential(*modules)
    ft_backbone.out_channels = out_channels
    return ft_backbone


# Use this when you have Adaptive Pooling layer in End.
# When Model.features is not applicable.
def _create_backbone_adaptive(model: nn.Module, out_channels: int = None):
    """
    Creates backbone by removing linear after Adaptive Pooling layer.
    Args:
        model: torch.nn model with adaptive pooling layer.
        out_channels (Optional) : Number of out_channels in last layer.
    """
    if out_channels is None:
        modules_total = list(model.children())
        out_channels = modules_total[-1].in_features
    return _create_backbone_generic(model, out_channels=out_channels)


def _create_backbone_features(model: nn.Module, out_channels: int):
    """
    Creates backbone from feature sequential block.
    Args:
        model: torch.nn model with features as sequential block.
        out_channels: Number of out_channels in last layer.
    """
    ft_backbone = model.features
    ft_backbone.out_channels = out_channels
    return ft_backbone


def create_torchvision_backbone(model_name: str, pretrained: str = None):
    """
    Creates CNN backbone from Torchvision.
    Args:
        model_name (str) : Name of the model. E.g. resnet18
        pretrained (str) : Pretrained weights dataset "imagenet", etc
    """

    model_selected = TORCHVISION_MODEL_ZOO[model_name]
    net = model_selected(pretrained=False)

    if pretrained is not None:
        state_dict = _load_pretrained_weights(WEIGHTS_DICT, model_name, pretrained=pretrained)
        net.load_state_dict(state_dict)

    if model_name in MOBILENET_MODELS:
        out_channels = 1280
        ft_backbone = _create_backbone_features(net, out_channels)
        return ft_backbone, out_channels

    elif model_name in VGG_MODELS:
        out_channels = 512
        ft_backbone = _create_backbone_features(net, out_channels)
        return ft_backbone, out_channels

    elif model_name in RESNET_SMALL_MODELS:
        out_channels = 512
        ft_backbone = _create_backbone_adaptive(net, out_channels)
        return ft_backbone, out_channels

    elif model_name in RESNET_LARGE_MODELS:
        out_channels = 2048
        ft_backbone = _create_backbone_adaptive(net, 2048)
        return ft_backbone, out_channels

    elif model_name in MNASNET_MODELS:
        out_channels = 1280
        ft_backbone = _create_backbone_adaptive(net, 1280)
        return ft_backbone, out_channels

    else:
        raise ValueError(f"Unsupported model: '{model_name}'")
