"""
Quantization Aware Training with Quickvision
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T

import torch.quantization
from torch.quantization import convert
from torch.quantization import QuantStub, DeQuantStub
from quickvision.models.classification import cnn

# Install Quickvision
# pip install -q git+https://github.com/Quick-AI/quickvision.git


if __name__ == "__main__":

    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    # Create CIFAR10 Dataset and DataLoaders"
    train_dataset = torchvision.datasets.CIFAR10("./data", download=True, train=True, transform=train_transforms)
    valid_dataset = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=valid_transforms)

    TRAIN_BATCH_SIZE = 512  # Training Batch Size
    VALID_BATCH_SIZE = 512  # Validation Batch Size

    train_loader = torch.utils.data.DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, VALID_BATCH_SIZE, shuffle=False)

    # Create Quantization Aware Model

    qat_model = cnn.create_vision_cnn("mobilenet_v2", pretrained="imagenet", num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-3)

    # Set Quantization Configurations

    qat_model.config = torch.quantization.get_default_qat_qconfig("fbgemm")
    _ = torch.quantization.prepare_qat(qat_model, inplace=True)

    # We can fine-tune / train the qat_models on GPU too.

    for param in qat_model.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = qat_model.to(device)

    NUM_TRAIN_BATCHES = 5  # You can pass these too in train step if you want small subset to train
    NUM_VAL_BATCHES = 5  # You can pass these too in train step if you want small subset to validate

    # Train with Quickvision !

    history = cnn.fit(epochs=3, model=qat_model, train_loader=train_loader,
                      val_loader=valid_loader, criterion=criterion, device=device,
                      optimizer=optimizer)

    qat_model.cpu()  # We need to move to cpu for conversion.

    qat_model_trained = torch.quantization.convert(qat_model, inplace=False)
    print("Converted the Quantization aware training model.")
    # torch.save(model_quantized_and_trained.state_dict(), config.QAT_SAVE_PATH)
