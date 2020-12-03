# flake8: noqa

import torchvision

TORCHVISION_MODEL_ZOO = {
    "vgg11": torchvision.models.vgg11(pretrained=False, **kwargs),
    "vgg13": torchvision.models.vgg13(pretrained=False, **kwargs),
    "vgg16": torchvision.models.vgg16(pretrained=False, **kwargs),
    "vgg19": torchvision.models.vgg19(pretrained=False, **kwargs),
    "resnet18": torchvision.models.resnet18(pretrained=False, **kwargs),
    "resnet34": torchvision.models.resnet34(pretrained=False, **kwargs),
    "resnet50": torchvision.models.resnet50(pretrained=False, **kwargs),
    "resnet101": torchvision.models.resnet101(pretrained=False, **kwargs),
    "resnet152": torchvision.models.resnet152(pretrained=False, **kwargs),
    "resnext50_32x4d": torchvision.models.resnext50_32x4d(pretrained=False, **kwargs),
    "resnext101_32x8d": torchvision.models.resnext101_32x8d(pretrained=False, **kwargs),
    "mnasnet0_5": torchvision.models.mnasnet0_5(pretrained=False, **kwargs),
    "mnasnet0_75": torchvision.models.mnasnet0_75(pretrained=False, **kwargs),
    "mnasnet1_0": torchvision.models.mnasnet1_0(pretrained=False, **kwargs),
    "mnasnet1_3": torchvision.models.mnasnet1_3(pretrained=False, **kwargs),
    "mobilenet_v2": torchvision.models.mobilenet_v2(pretrained=False, **kwargs),
}
