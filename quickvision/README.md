# Quickvision !

- This is how it is structured: -

## Datasets (Unstable): -

- Provides `torchvision` datasets for common use cases and scenarios.

## Losses (Unstable): -

- Losses which are not currently supported by torch, but are popularly used.

## Models: -

- Implementation of Computer Vision Models for training as well as transfer learning.
- Here we leverage Torchvision models and models from PyTorch Hub as well.
- These are implemented with a PyTorch training API as well as consistent Lightning API.
- All have APIs such as `train_step`, `val_step`, `fit` and sanity APIs.
- All support Mixed Precision Training.

## Optimizers (Unstable): -

- Commonly used Optimizers pertaining to computer vision.

## Pretrained: -

- A list of pretrained models and weights which we use.
- You can contribute your models trained on different datasets here !

## Utils (Unstable): -

- Commonly used utilities.
