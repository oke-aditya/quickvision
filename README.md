# Quickvision

- Faster Computer Vision.

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/issues)
[![GitHub forks](https://img.shields.io/github/forks/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/network)
[![GitHub stars](https://img.shields.io/github/stars/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/stargazers)
[![GitHub license](https://img.shields.io/github/license/Quick-AI/quickvision)](https://github.com/Quick-AI/quickvision/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/Quick-AI/quickvision/branch/master/graph/badge.svg?token=VAFPQTQK1I)](https://codecov.io/gh/Quick-AI/quickvision)

![PEP8](https://github.com/Quick-AI/quickvision/workflows/Check%20Code%20formatting/badge.svg)
![CI Tests](https://github.com/Quick-AI/quickvision/workflows/CI%20Tests/badge.svg)
![Docs](https://github.com/Quick-AI/quickvision/workflows/Deploy%20mkdocs/badge.svg)
![PyPi Release](https://github.com/Quick-AI/quickvision/workflows/PyPi%20Release/badge.svg)

[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/quickai/shared_invite/zt-iz7tqk3r-IQa4SoxJGIK5WS8VdZhzeQ)
[![Downloads](https://pepy.tech/badge/quickvision)](https://pepy.tech/project/quickvision)
[![Downloads](https://pepy.tech/badge/quickvision/month)](https://pepy.tech/project/quickvision)
[![Downloads](https://pepy.tech/badge/quickvision/week)](https://pepy.tech/project/quickvision)

</div>

![demo](/assets/demo.png)

### Install Quickvision

- Install from PyPi.
- Current stable `release 0.1.1` needs `PyTorch 1.7.1` and `torchvision 0.8.2`.

    ```
    pip install quickvision
    ```

## What is Quickvision?

- Quickvision makes Computer Vision tasks much faster and easier with PyTorch.

    It provides: -

    1. Easy to use PyTorch native API, for `fit()`, `train_step()`, `val_step()` of models.
    2. Easily customizable and configurable models with various backbones.
    3. A complete PyTorch native interface. All models are `nn.Module`, all the training APIs are optional and not binded to models.
    4. A lightning API which helps to accelerate training over multiple GPUs, TPUs.
    5. A datasets API to convert common data formats very easily and quickly to PyTorch formats.
    6. A minimal package, with very low dependencies.

- Train your models faster. Quickvision has already implemented the long learning in PyTorch.

## Quickvision is just PyTorch!!

- Quickvision does not make you learn a new library. If you know PyTorch, you are good to go!!!
- Quickvision does not abstract any code from PyTorch, nor implements any custom classes over it.
- It keeps the data format in `Tensor` so that you don't need to convert it.

### Do you want just a model with some backbone configuration?

- Use model made by us. It's just a `nn.Module` which has Tensors only Input and Output format.
- Quickvision provides reference scripts too for training it!

### Do you want to train your model but not write lengthy loops?

- Just use our training methods such as `fit()`, `train_step()`, `val_step()`.

### Do you want multi GPU training but worried about model configuration?

- Just subclass the PyTorch Lightning model! 
- Implement the `train_step()`, `val_step()`.
