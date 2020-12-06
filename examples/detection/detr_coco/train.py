import os
import numpy as np
from argparse import ArgumentParser
import torch
import torch.utils.data
from pprint import pprint
from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import pytorch_lightning as pl
from quickvision.models.detection.detr import lit_detr as lit_obj_det
from coco import build as build_dataset
from pytorch_lightning.loggers import WandbLogger


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--coco_path", type=str)
    parser.add_argument(
        "--masks", action="store_true", help="Train segmentation head if the flag is provided"
    )
    args = parser.parse_args()
    args.coco_path = os.path.expandvars(args.coco_path)
    return args


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    args = get_args()
    dataset = build_dataset("train", args)
    dataset_test = build_dataset("val", args)
    num_workers = 4
    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    num_classes = 91
    model = lit_obj_det(
        learning_rate=1e-3,
        num_classes=num_classes,
        backbone="resnet50",
        fpn=True,
        pretrained_backbone="imagenet",
    )

    wandb_logger = WandbLogger(project="coco", tags=["quickvision", "detr", "imagenet"])
    trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=wandb_logger)
    trainer.fit(model, train_loader, test_loader)

