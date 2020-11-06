import pytorch_lightning as pl
import torch
import torchvision
from vision.models.classification.cnn.lightning_trainer import CNN
import config


def create_cifar10_dataset(train_transforms, valid_transforms):
    """Creates CIFAR10 train dataset and a test dataset.
    Args:
    train_transforms: Transforms to be applied to train dataset.
    test_transforms: Transforms to be applied to test dataset.
    """
    # This code can be re-used for other torchvision Image Dataset too.
    train_set = torchvision.datasets.CIFAR10("./data", download=True, train=True, transform=train_transforms)
    valid_set = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=valid_transforms)

    return train_set, valid_set


def create_loaders(train_dataset, valid_dataset,
                   train_batch_size=32, valid_batch_size=32, num_workers=4, **kwargs):

    """
    Creates train loader and test loader from train and test datasets
    Args:
        train_dataset: Torchvision train dataset.
        valid_dataset: Torchvision valid dataset.
        train_batch_size (int) : Default 32, Training Batch size
        valid_batch_size (int) : Default 32, Validation Batch size
        num_workers (int) : Defualt 1, Number of workers for training and validation.
    """

    train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, shuffle=True,
                                               num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, valid_batch_size, shuffle=False,
                                               num_workers=num_workers)
    return train_loader, valid_loader


if __name__ == "__main__":
    pl.seed_everything(seed=42)
    # print(model)
    print("Creating Train and Validation Dataset")
    train_set, valid_set = create_cifar10_dataset(config.train_transforms, config.valid_transforms)
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = create_loaders(train_set, train_set, config.TRAIN_BATCH_SIZE,
                                                config.VALID_BATCH_SIZE, config.NUM_WORKERS,)

    print("Train and Validation Dataloaders Created")

    print("Creating Model")
    model = CNN("resnet18", num_classes=10, pretrained=True)
    # print(model)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_loader, valid_loader)
