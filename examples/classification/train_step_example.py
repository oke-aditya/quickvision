import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from vision.models.classification.cnn import model_factory
import config
from vision.models import model_utils
from vision.models.classification import engine


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

    MODEL_NAME = "resnet18"
    NUM_ClASSES = 10
    IN_CHANNELS = 3
    PRETRAINED = True  # If True -> Fine Tuning else Scratch Training
    EPOCHS = 5
    EARLY_STOPPING = True  # If you need early stoppoing for validation loss
    SAVE_PATH = f"{MODEL_NAME}.pt"
    SEED = 42

    # Train and validation Transforms which you would like
    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    model_utils.seed_everything(SEED)
    print(f"Setting Seed for the run, seed = {config.SEED}")

    print("Creating Train and Validation Dataset")
    train_set, valid_set = create_cifar10_dataset(train_transforms, valid_transforms)
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = create_loaders(train_set, train_set)
    print("Train and Validation Dataloaders Created")

    print("Creating Model")
    model = model_factory.create_torchvision_model(MODEL_NAME, num_classes=NUM_ClASSES, pretrained=True)

    if torch.cuda.is_available():
        print("Model Created. Moving it to CUDA")
    else:
        print("Model Created. Training on CPU only")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = (nn.CrossEntropyLoss())  # All classification problems we need Cross entropy loss
    early_stopper = model_utils.EarlyStopping(patience=7, verbose=True, path=SAVE_PATH)

    for epoch in tqdm(range(EPOCHS)):
        print()
        print(f"Training Epoch = {epoch}")
        train_metrics = engine.train_step(model, train_loader, criterion, device, optimizer)
        print()

        print(f"Validating Epoch = {epoch}")
        valid_metrics = engine.val_step(model, valid_loader, criterion, device)

        validation_loss = valid_metrics["loss"]
        early_stopper(validation_loss, model=model)

        if early_stopper.early_stop:
            print("Saving Model and Early Stopping")
            print("Early Stopping. Ran out of Patience for validation loss")
            break

        print("Done Training, Model Saved to Disk")
