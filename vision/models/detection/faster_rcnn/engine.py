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

    """
    Performs one step of training. Calculates loss, forward pass, computes gradient and returns metrics.
    Args:
        model : PyTorch Faster RCNN Model.
        train_loader : Train loader.
        device : "cuda" or "cpu"
        optimizer : Torch optimizer to train.
        scheduler : Learning rate scheduler.
        num_batches : (optional) Integer To limit training to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        grad_penalty : (optional) To penalize with l2 norm for big gradients.
        scaler: (optional)  Pass torch.cuda.amp.GradScaler() for fp16 precision Training.
    """

    start_train_step = time.time()

    model.train()
    last_idx = len(train_loader) - 1
    batch_time_m = model_utils.AverageMeter()
    cnt = 0
    batch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        images = list(image.to(device) for image in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # zero the parameter gradients
        optimizer.zero_grad()
        if scaler is not None:
            with amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_v for loss_v in loss_dict.values())
                scaler.scale(loss).backward()
                # Step using scaler.step()
                scaler.step(optimizer)
                # Update for next iteration
                scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_v for loss_v in loss_dict.values())
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        cnt += 1
        batch_time_m.update(time.time() - batch_start)
        batch_start = time.time()
        if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
            print("Batch Train Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  ".format(
                  batch_time=batch_time_m,))
#   "loss classifier: {loss_d.loss_classifier:>7.4f} "
#   "loss box_reg:  {loss_d.loss_box_reg:>7.4f} "
#   "loss objectness: {loss_d.loss_objectness:>7.4f} "
#   "loss rpn box reg: {loss_d.loss_rpn_box_reg:>7.4f}"

        if num_batches is not None:
            if cnt >= num_batches:
                print(f"Done till {num_batches} train batches")
                end_train_step = time.time()
                print(f"Time taken for Training step = {end_train_step - start_train_step} sec")
                return loss_dict

    end_train_step = time.time()
    print(f"Time taken for Training step = {end_train_step - start_train_step} sec")
    return loss_dict


def val_step(model, val_loader, device, num_batches=None,
             log_interval: int = 100):

    """
    Performs one step of validation. Calculates loss, forward pass and returns metrics.
    Args:
        model : PyTorch FasterRCNN Model.
        val_loader : Validation loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    start_val_step = time.time()
    last_idx = len(val_loader) - 1
    batch_time_m = model_utils.AverageMeter()
    cnt = 0
    model.eval()
    batch_start = time.time()
    metrics = OrderedDict()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(images)
            iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, out)]).mean()
            giou = torch.stack([_evaluate_giou(t, o) for t, o in zip(targets, out)]).mean()
            metrics["iou"] = iou
            metrics["giou"] = giou

            cnt += 1
            batch_time_m.update(time.time() - batch_start)
            batch_start = time.time()

            if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
                print("Batch Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  ".format(
                      batch_time=batch_time_m,))

            if num_batches is not None:
                if cnt >= num_batches:
                    print(f"Done till {num_batches} Validation batches")
                    end_val_step = time.time()
                    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
                    return metrics

    end_val_step = time.time()
    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
    return metrics


def fit(model, epochs, train_loader, val_loader,
        device, optimizer, scheduler=None,
        num_batches: int = None, log_interval: int = 100,
        fp16: bool = False, ):

    """
    A fit function that performs training for certain number of epochs.
    Args:
        model : A pytorch Faster RCNN Model.
        epochs: Number of epochs to train.
        train_loader : Train loader.
        val_loader : Validation loader.
        device : "cuda" or "cpu"
        optimizer : PyTorch optimizer.
        scheduler : (optional) Learning Rate scheduler.
        early_stopper: (optional) A model_utils provided early stopper, based on validation loss.
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        fp16 : (optional) To use Mixed Precision Training using float16 dtype.
    """
    history = {}
    train_loss = []
    val_iou = []
    val_giou = []

    if fp16 is True:
        print("Training with Mixed precision fp16 scaler")
        scaler = amp.GradScaler()
    else:
        scaler = None

    for epoch in tqdm(range(epochs)):
        print()
        print(f"Training Epoch = {epoch}")
        loss_dict = train_step(model, train_loader, device, optimizer,
                               scheduler, num_batches, log_interval, scaler)
        metrics = val_step(model, val_loader, device, num_batches, log_interval)

        # Possibly we can use individual losses
        loss = sum(loss_v for loss_v in loss_dict.values())
        train_loss.append(loss.detach())

        avg_iou = torch.stack([metrics["iou"]]).mean()
        avg_giou = torch.stack([metrics["giou"]]).mean()

        val_iou.append(avg_iou)
        val_giou.append(avg_giou)

    history = {"train": {"train_loss": train_loss},
               "val": {"val_iou": val_iou, "val_giou": val_giou}}

    return history


def train_sanity_fit(model, train_loader,
                     device, num_batches: int = None, log_interval: int = 100,
                     fp16: bool = False,):

    """
    Performs Sanity fit over train loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and val_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch Model.
        train_loader : Train loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches.
                                 Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        fp16: : (optional) If True uses PyTorch native mixed precision Training.
    """
    pass


def val_sanity_fit(model, val_loader,
                   device, num_batches: int = None,
                   log_interval: int = 100,):

    """
    Performs Sanity fit over valid loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and val_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch CNN Model.
        val_loader : Validation loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches.
                                 Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    pass


def sanity_fit(model, train_loader, val_loader,
               device, num_batches: int = None,
               log_interval: int = 100, fp16: bool = False,):

    """
    Performs Sanity fit over train loader and valid loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and val_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch CNN Model.
        train_loader : Training loader.
        val_loader : Validation loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches.
                                 Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """

    pass
