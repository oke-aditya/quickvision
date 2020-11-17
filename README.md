# Quickvision

- Faster Computer Vision.

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/issues)
[![GitHub forks](https://img.shields.io/github/forks/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/network)
[![GitHub stars](https://img.shields.io/github/stars/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/stargazers)
[![GitHub license](https://img.shields.io/github/license/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision)
[![codecov](https://codecov.io/gh/Quick-AI/quickvision/branch/master/graph/badge.svg?token=VAFPQTQK1I)](https://codecov.io/gh/Quick-AI/quickvision)

![PEP8](https://github.com/Quick-AI/quickvision/workflows/Check%20Code%20formatting/badge.svg)
![CI Tests](https://github.com/Quick-AI/quickvision/workflows/CI%20Tests/badge.svg)
![Docs](https://github.com/Quick-AI/quickvision/workflows/Deploy%20mkdocs/badge.svg)
![PyPi Release](https://github.com/Quick-AI/quickvision/workflows/PyPi%20Release/badge.svg)

</div>

![demo](/assets/demo.png)

### Install Quickvision

- Install directly from GitHub. Very soon it will be available over PyPi.

```
pip install -q git+https://github.com/Quick-AI/quickvision.git
```

## What is Quickvision?

- Quickvision makes Computer Vision tasks much faster and easier with PyTorch.

It provides: -

1. Easy to use torch native API, for `fit()`, `train_step()`, `val_step()` of models.
2. Easily customizable and configurable models with various backbones.
3. A complete torch native interface. All models are `nn.Module` all the training APIs are optional and not binded to models.
4. A lightning API which helps to accelerate training over multiple GPUs, TPUs.
5. A datasets API to common data format very easily and quickly to torch formats.
6. A minimal package, with very low dependencies.

- Train your models faster. Quickvision has already implmented the long learning in torch.

## Quickvision is just Torch!!

- Quickvision does not make you learn a new library. If you know PyTorch you are good to go!!!
- Quickvision does not abstract any code from torch, nor implements any custom classes over it.
- It keeps the data format in `Tensor` only. You don't need to convert it.

### Do you want just a model with some backbone configuration?

- Use model made by us. It's just a `nn.Module` which has Tensors only Input and Output format.
- Quickvision provides reference scripts too for training it!

### Do you want to train your model but not write lengthy loops?

- Just use our training methods such as `fit()`, `train_step()`, `val_step()`.

### Do you want multi GPU training but worried about model configuration?

- Just subclass the PyTorch Lightning model! 
- Implement the `train_step`, `val_step`.
