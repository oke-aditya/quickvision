import torch.nn as nn
import torchvision

__all__ = ["create_torchvision_backbone"]


def _create_backbone_generic(model, out_channels: int):
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
def _create_backbone_adaptive(model, out_channels: int = None):
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


def _create_backbone_features(model, out_channels: int):
    """
    Creates backbone from feature sequential block.
    Args:
        model: torch.nn model with features as sequential block.
        out_channels: Number of out_channels in last layer.
    """
    ft_backbone = model.features
    ft_backbone.out_channels = out_channels
    return ft_backbone


def create_torchvision_backbone(model_name: str, pretrained: bool = True):
    """
    Creates CNN backbone from Torchvision.
    Args:
        model_name (str) : Name of the model. E.g. resnet18
        pretrained (bool) : If true uses modelwweights pretrained on ImageNet.
    """
    if model_name == "mobilenet_v2":
        mobile_net = torchvision.models.mobilenet_v2(pretrained)
        out_channels = 1280
        ft_backbone = _create_backbone_features(mobile_net, 1280)
        return ft_backbone, out_channels

    elif model_name in ["vgg11", "vgg13", "vgg16", "vgg19", ]:
        out_channels = 512
        if model_name == "vgg11":
            vgg_net = torchvision.models.vgg11(pretrained)
        elif model_name == "vgg13":
            vgg_net = torchvision.models.vgg13(pretrained)
        elif model_name == "vgg16":
            vgg_net = torchvision.models.vgg16(pretrained)
        elif model_name == "vgg19":
            vgg_net = torchvision.models.vgg19(pretrained)

        ft_backbone = _create_backbone_features(vgg_net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet18", "resnet34"]:
        out_channels = 512
        if model_name == "resnet18":
            resnet_net = torchvision.models.resnet18(pretrained)
        elif model_name == "resnet34":
            resnet_net = torchvision.models.resnet34(pretrained)

        ft_backbone = _create_backbone_adaptive(resnet_net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", ]:
        out_channels = 2048
        if model_name == "resnet50":
            resnet_net = torchvision.models.resnet50(pretrained)
        elif model_name == "resnet101":
            resnet_net = torchvision.models.resnet101(pretrained)
        elif model_name == "resnet152":
            resnet_net = torchvision.models.resnet152(pretrained)
        elif model_name == "resnext50_32x4d":
            resnet_net = torchvision.models.resnext50_32x4d(pretrained)
        elif model_name == "resnext101_32x8d":
            resnet_net = torchvision.models.resnext101_32x8d(pretrained)

        ft_backbone = _create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone, out_channels

    elif model_name in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        out_channels = 1280
        if model_name == "mnasnet0_5":
            mnasnet_net = torchvision.models.mnasnet0_5(pretrained)
        elif model_name == "mnasnet0_75":
            mnasnet_net = torchvision.models.mnasnet0_75(pretrained)
        elif model_name == "mnasnet1_0":
            mnasnet_net = torchvision.models.mnasnet1_0(pretrained)
        elif model_name == "mnasnet1_3":
            mnasnet_net = torchvision.models.mnasnet1_3(pretrained)

        ft_backbone = _create_backbone_adaptive(mnasnet_net, 1280)
        return ft_backbone, out_channels

    else:
        raise ValueError("No such model implemented in torchvision")
