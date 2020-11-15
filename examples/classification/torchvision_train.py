import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from quickvision.models.classification import cnn
import config
from quickvision import utils


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Setting Seed for the run, seed = {config.SEED}")
    utils.seed_everything(config.SEED)

    print("Creating Train and Validation Dataset")
    train_set, valid_set = create_cifar10_dataset(config.train_transforms, config.valid_transforms)
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = create_loaders(train_set, train_set, train_batch_size=32, valid_batch_size=32,)

    print("Train and Validation Dataloaders Created")
    print("Creating Model")

    # model = model_factory.create_timm_model(config.MODEL_NAME, num_classes=config.NUM_ClASSES,
    #                                         in_channels=config.IN_CHANNELS, pretrained=config.PRETRAINED,)

    model = cnn.create_vision_cnn("resnet50", num_classes=10, pretrained="imagenet",)

    if torch.cuda.is_available():
        print("Model Created. Moving it to CUDA")
        device = "cuda"
    else:
        print("Model Created. Training on CPU only")
        device = "cpu"

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Optionially a schedulear
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-4, max_lr=1e-3, mode="min")

    criterion = nn.CrossEntropyLoss()  # All classification problems we need Cross entropy loss

    early_stopper = utils.EarlyStopping(patience=7, verbose=True, path=config.SAVE_PATH)

    history = cnn.fit(model=model, epochs=10, train_loader=train_loader,
                      valid_loader=valid_loader, criterion=criterion, device=device,
                      optimizer=optimizer, early_stopper=early_stopper,)

    print("Done !!")
