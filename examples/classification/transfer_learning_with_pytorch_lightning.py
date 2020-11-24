import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from quickvision.models.classification.cnn import lit_cnn

TRAIN_BATCH_SIZE = 512  # Training Batch Size
VALID_BATCH_SIZE = 512  # Validation Batch Size


if __name__ == "__main__":
    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.CIFAR10("./data", download=True, train=True, transform=train_transforms)
    valid_dataset = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, VALID_BATCH_SIZE, shuffle=False)

    # Create a model with pretrained imagenet weights

    model_imagenet = lit_cnn("resnet18", num_classes=10, pretrained="imagenet")

    # You can pass all the possible Trainer Arguments.

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(model_imagenet, train_loader, valid_loader)

    # Training without any pretrained weights.

    model_ssl = lit_cnn("resnet18", num_classes=10, pretrained=None)

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(model_ssl, train_loader, valid_loader)

    # Custom Training with Lightning !

    # To write your own Training logic, metrics, logging. Subclass the `lit_cnn` and write your own logic !

    class CustomTraining(lit_cnn):
        def training_step(self, batch, batch_idx):
            images, targets = batch
            outputs = self.forward(images)
            train_loss = F.cross_entropy(outputs, targets, reduction='sum')
            # Possible we can compute top-1 and top-5 accuracy here.
            return {"loss": train_loss}

        def validation_step(self, batch, batch_idx):
            images, targets = batch
            outputs = self.forward(images)
            val_loss = F.cross_entropy(outputs, targets, reduction='sum')
            # Possible we can compute top-1 and top-5 accuracy here.
            return {"loss": val_loss}

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Create Model provided by Quickvision !

    model_imagenet = CustomTraining("resnet18", num_classes=10, pretrained="imagenet")

    # Train with PL Trainer

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(model_imagenet, train_loader, valid_loader)
