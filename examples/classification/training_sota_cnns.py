# -*- coding: utf-8 -*-
# Training SOTA CNNs

# Quickvision + PyTorch Image Models CNNs

# - A Huge home to SOTA CNNs is
# [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) by Ross Wightman.

# - It has high quality models with ImageNet weights.

# - Let's train these for Trasnfer Learning Tasks !


# Install PyTorch Image Models
# ! pip install -q timm
# ! pip install -q git+https://github.com/Quick-AI/quickvision.git

import torch
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# - A List of PreTrained Models Provided By Timm

import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)

# - Let's stick to EfficientNet and Train it !

# CIFAR10 Dataset and Data Loader

# - We use CIFAR10 Dataset to train on.
# - It is directly available in torchvision


if __name__ == "__main__":

    TRAIN_BATCH_SIZE = 512  # Training Batch Size
    VALID_BATCH_SIZE = 512  # Validation Batch Size

    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.CIFAR10("./data", download=True, train=True, transform=train_transforms)
    valid_dataset = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, VALID_BATCH_SIZE, shuffle=False)
    VALID_BATCH_SIZE = 512  # Validation Batch Size

    # Train EfficientNet from PyTorch Image Models

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=10, in_chans=3)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Here is where you can use Quickvision's Training Recipe to train these models

    from quickvision.models.classification import cnn

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    history = cnn.fit(model=model, epochs=2, train_loader=train_loader,
                      val_loader=valid_loader, criterion=criterion, device=device,
                      optimizer=optimizer)

    # - In return you get a Keras Like History Dictionary.
    # - It keeps a track of Metrics

    pprint(history)

    # - You can also get a granular control over these using `train_step`, `val_step` methods.

    EPOCHS = 2

    for epoch in tqdm(range(EPOCHS)):
        print()
        print(f"Training Epoch = {epoch}")
        train_metrics = cnn.train_step(model, train_loader, criterion, device, optimizer)
        print()

        print(f"Validating Epoch = {epoch}")
        valid_metrics = cnn.val_step(model, valid_loader, criterion, device)

    # - Again, quickvision computes metrics for you !
    # - You can print them to have look !

    pprint(train_metrics)

    pprint(valid_metrics)
