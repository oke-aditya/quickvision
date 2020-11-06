import torchvision
import torch
import torchvision.transforms as T
from PIL import Image

__all__ = ["im2tensor", "to_np", "requires_gradient"]


def im2tensor(image):
    aug = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.5,), (0.5,))])
    tensor = aug(image)
    tensor = torch.unsqueeze(tensor, 0)
    return tensor


def to_np(t):
    return t.detach().cpu().numpy()


def requires_gradient(model, layer):
    return list(model.parameters())[layer].requires_grad
