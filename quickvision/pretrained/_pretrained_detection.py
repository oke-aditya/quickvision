# flake8: noqa

frcnn_weights_dict = {
    "resnet50": {
        "coco": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    },
}

retina_weights_dict = {
    "resnet50": {
        "coco": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth"
    },
}

detr_weights_dict = {
    "resnet50": {
        "coco": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
    },
    "resnet101": {
        "coco": "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth"
    },
    "resnet50_dc5": {
        "coco": "ttps://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth"
    },
    "resnet101_dc5": {
        "coco": "https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth"
    },
}
