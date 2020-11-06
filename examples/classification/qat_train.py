# Just an example of how quantization aware training can be used with the same engine.

import timm
import config
from vision.models.classification.cnn import model_factory
from vision.models.classification.cnn import engine
from tqdm import tqdm
import torch.nn as nn
from vision.models import model_utils
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import torch.optim as optim
from torch.quantization import convert
import torchvision


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

    train_ds, valid_ds = create_cifar10_dataset(config.train_transforms, config.valid_transforms)

    train_loader, valid_loader = create_loaders(train_ds, valid_ds)

    qat_model = model_factory.create_timm_cnn(config.MODEL_NAME, config.NUM_ClASSES,
                                              config.IN_CHANNELS, config.PRETRAINED)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-3)
    early_stopper = model_utils.EarlyStopping(patience=3, verbose=True, path=config.SAVE_PATH)

    qat_model.config = torch.quantization.get_default_qat_qconfig("fbgemm")
    _ = torch.quantization.prepare_qat(qat_model, inplace=True)

    # We can fine-tune / train the qat_models on GPU too.

    for param in qat_model.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = qat_model.to(device)

    # NUM_TRAIN_BATCHES = 5 You can pass these too in train step if you want small subset to train
    # NUM_VAL_BATCHES = 5  You can pass these too in train step if you want small subset to validate

    history = engine.fit(epochs=config.EPOCHS, model=qat_model, train_loader=train_loader,
                         valid_loader=valid_loader, criterion=criterion, device=device,
                         optimizer=optimizer, early_stopper=early_stopper,)

    qat_model.cpu()  # We need to move to cpu for conversion.

    qat_model_trained = convert(qat_model, inplace=False)
    print("Converted the Quantization aware training model.")
    # torch.save(model_quantized_and_trained.state_dict(), config.QAT_SAVE_PATH)
