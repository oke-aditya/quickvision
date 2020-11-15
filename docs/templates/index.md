# Quickvision

- Faster Computer Vision

## What is Quickvision ??

- Quickvision makes Computer Vision tasks much faster and easier with PyTorch.

It provides: -

1. Easy to use torch native API, for `fit()`, `train_step()`, `val_step()` of models.
2. Easily customizable and configurable models with various backbones.
3. A complete torch native interface. All models are `nn.Module` all the training APIs are optional and not binded to models.
4. A lightning API which helps to accelerate training over multiple GPUs, TPUs.
5. A datasets API to common data format very easily and quickly to torch formats.
6. A minimal package, with very low dependencies.

- Train your models faster. Quickvision has already implmented the long learning in torch.

## Quickvision is just Torch !!

- Quickvision does not make you learn a new library. If you know PyTorch you are good to go !!!
- Quickvision does not abstract any code from torch, nor implements any custom classes over it.
- It keeps the data format in `Tensor` only. You don't need to convert it.

### Do you want just a model with some backbone configuration ?

- Use model made by us. It's just a `nn.Module` which has Tensors only Input and Output format.
- Quickvision provides reference scripts too for training it !.

### Do You wan't to train your model but not write lengthy loops ?

- Just use our training methods such as `fit()`, `train_step()`, `val_step()`.

### Do You wan't multi GPU training. But are worried about model configuration ?

- Just Subclass the PyTorch Lightning model ! 
- Implement the `train_step`, `val_step`
