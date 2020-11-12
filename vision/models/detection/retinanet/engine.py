import torch
from torch.cuda import amp
from vision.models import model_utils
from tqdm import tqdm
import time
from collections import OrderedDict
from vision.models.detection.utils import _evaluate_iou, _evaluate_giou

__all__ = ["train_step", "val_step", "fit", "train_sanity_fit",
           "val_sanity_fit", "sanity_fit", ]


def train_step(model, train_loader, device, optimizer,
               scheduler=None, num_batches: int = None,
               log_interval: int = 100, scaler=None,):

    pass


def val_step(model, val_loader, device, num_batches=None,
             log_interval: int = 100):

    pass


def fit(model, epochs, train_loader, val_loader,
        device, optimizer, scheduler=None,
        num_batches: int = None, log_interval: int = 100,
        fp16: bool = False, ):
    pass


def train_sanity_fit(model, train_loader,
                     device, num_batches: int = None, log_interval: int = 100,
                     fp16: bool = False,):
    pass


def val_sanity_fit(model, val_loader,
                   device, num_batches: int = None,
                   log_interval: int = 100,):
    pass


def sanity_fit(model, train_loader, val_loader,
               device, num_batches: int = None,
               log_interval: int = 100, fp16: bool = False,):
    pass
