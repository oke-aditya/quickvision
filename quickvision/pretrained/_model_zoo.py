# flake8: noqa

import torchvision

TORCHVISION_MODEL_ZOO = {
    "vgg11": torchvision.models.vgg11(pretrained=False,),
    "vgg13": torchvision.models.vgg13(pretrained=False,),
    "vgg16": torchvision.models.vgg16(pretrained=False,),
    "vgg19": torchvision.models.vgg19(pretrained=False,),
    "resnet18": torchvision.models.resnet18(pretrained=False,),
    "resnet34": torchvision.models.resnet34(pretrained=False,),
    "resnet50": torchvision.models.resnet50(pretrained=False,),
    "resnet101": torchvision.models.resnet101(pretrained=False,),
    "resnet152": torchvision.models.resnet152(pretrained=False,),
    "resnext50_32x4d": torchvision.models.resnext50_32x4d(pretrained=False,),
    "resnext101_32x8d": torchvision.models.resnext101_32x8d(pretrained=False,),
    "mnasnet0_5": torchvision.models.mnasnet0_5(pretrained=False,),
    "mnasnet0_75": torchvision.models.mnasnet0_75(pretrained=False,),
    "mnasnet1_0": torchvision.models.mnasnet1_0(pretrained=False,),
    "mnasnet1_3": torchvision.models.mnasnet1_3(pretrained=False,),
    "mobilenet_v2": torchvision.models.mobilenet_v2(pretrained=False,),
    "wide_resnet50_2": torchvision.models.wide_resnet50_2(pretrained=False,),
    "wide_resnet101_2": torchvision.models.wide_resnet101_2(pretrained=False,),
}
