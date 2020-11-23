import torch
from torch import nn, Tensor
from torch.cuda import amp
from quickvision import utils
from tqdm import tqdm
import time
from collections import OrderedDict
from quickvision.models.detection.utils import _evaluate_iou, _evaluate_giou


__all__ = ["train_step", "val_step", "fit", "train_sanity_fit",
           "val_sanity_fit", "sanity_fit", ]


def train_step(model: nn.Module, train_loader, device: str, optimizer,
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

    model = model.to(device)
    start_train_step = time.time()

    model.train()
    last_idx = len(train_loader) - 1
    batch_time_m = utils.AverageMeter()
    cnt = 0
    batch_start = time.time()
    metrics = OrderedDict()

    total_loss = utils.AverageMeter()
    loss_classifier = utils.AverageMeter()
    loss_box_reg = utils.AverageMeter()
    loss_objectness = utils.AverageMeter()
    loss_rpn_box_reg = utils.AverageMeter()

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

        total_loss.update(loss.item())
        loss_classifier.update(loss_dict["loss_classifier"].item())
        loss_box_reg.update(loss_dict["loss_box_reg"].item())
        loss_objectness.update(loss_dict["loss_objectness"].item())
        loss_rpn_box_reg.update(loss_dict["loss_rpn_box_reg"].item())

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
                end_train_step = time.time()
                metrics["total_loss"] = total_loss.avg
                metrics["loss_classifier"] = loss_classifier.avg
                metrics["loss_box_reg"] = loss_box_reg.avg
                metrics["loss_objectness"] = loss_objectness.avg
                metrics["loss_rpn_box_reg"] = loss_rpn_box_reg.avg

                print(f"Done till {num_batches} train batches")
                print(f"Time taken for Training step = {end_train_step - start_train_step} sec")
                return metrics

    end_train_step = time.time()
    metrics["total_loss"] = total_loss.avg
    metrics["loss_classifier"] = loss_classifier.avg
    metrics["loss_box_reg"] = loss_box_reg.avg
    metrics["loss_objectness"] = loss_objectness.avg
    metrics["loss_rpn_box_reg"] = loss_rpn_box_reg.avg
    print(f"Time taken for Training step = {end_train_step - start_train_step} sec")
    return metrics


def val_step(model: nn.Module, val_loader, device: str, num_batches: int = None,
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

    model = model.to(device)
    start_val_step = time.time()
    last_idx = len(val_loader) - 1
    batch_time_m = utils.AverageMeter()
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

            cnt += 1
            batch_time_m.update(time.time() - batch_start)
            batch_start = time.time()

            if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
                print("Batch Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  ".format(
                      batch_time=batch_time_m,))

            if num_batches is not None:
                if cnt >= num_batches:
                    avg_iou = torch.stack([iou]).mean()
                    avg_giou = torch.stack([giou]).mean()
                    metrics["iou"] = avg_iou
                    metrics["giou"] = avg_giou
                    print(f"Done till {num_batches} Validation batches")
                    end_val_step = time.time()
                    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
                    return metrics

    avg_iou = torch.stack([iou]).mean()
    avg_giou = torch.stack([giou]).mean()
    metrics["iou"] = avg_iou
    metrics["giou"] = avg_giou

    end_val_step = time.time()
    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
    return metrics


def fit(model: nn.Module, epochs: int, train_loader, val_loader,
        device: str, optimizer, scheduler=None,
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
        early_stopper: (optional) A utils provided early stopper, based on validation loss.
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
        train_metrics = train_step(model, train_loader, device, optimizer,
                                   scheduler, num_batches, log_interval, scaler)
        val_metrics = val_step(model, val_loader, device, num_batches, log_interval)

        # Possibly we can use individual losses
        train_loss.append(train_metrics["total_loss"])

        avg_iou = val_metrics["iou"]
        avg_giou = val_metrics["giou"]

        val_iou.append(avg_iou)
        val_giou.append(avg_giou)

    history = {"train": {"train_loss": train_loss},
               "val": {"val_iou": val_iou, "val_giou": val_giou}}

    return history


def train_sanity_fit(model: nn.Module, train_loader,
                     device: str, num_batches: int = None, log_interval: int = 100,
                     fp16: bool = False,):

    """
    Performs Sanity fit over train loader.
    Use this to dummy check your train_step function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch Faster RCNN Model.
        train_loader : Train loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches.
                                 Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        fp16: : (optional) If True uses PyTorch native mixed precision Training.
    """

    model = model.to(device)
    model.train()

    cnt = 0
    last_idx = len(train_loader) - 1
    train_sanity_start = time.time()

    if fp16 is True:
        scaler = amp.GradScaler()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        images = list(image.to(device) for image in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if fp16 is True:
            with amp.autocast():
                loss_dict = model(images, targets)
        else:
            loss_dict = model(images, targets)

        cnt += 1

        if last_batch or batch_idx % log_interval == 0:
            print(f"Train sanity check passed for batch till {batch_idx} batches")

        if num_batches is not None:
            if cnt >= num_batches:
                print(f"Done till {num_batches} train batches")
                print("All specified batches done")
                train_sanity_end = time.time()
                print(f"Train sanity fit check passed in time {train_sanity_end-train_sanity_start}")
                return True

    train_sanity_end = time.time()

    print("All specified batches done")
    print(f"Train sanity fit check passed in time {train_sanity_end-train_sanity_start}")

    return True


def val_sanity_fit(model: nn.Module, val_loader,
                   device: str, num_batches: int = None,
                   log_interval: int = 100,):

    """
    Performs Sanity fit over valid loader.
    Use this to dummy check your val_step function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and val_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch Faster RCNN Model.
        val_loader : Validation loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches.
                                 Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    model = model.to(device)
    model.eval()

    cnt = 0
    val_sanity_start = time.time()
    last_idx = len(val_loader) - 1

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            out = model(images)
            cnt += 1

            if last_batch or (batch_idx % log_interval) == 0:
                print(f"Val sanity check passed for batch till {batch_idx} batches")

            if num_batches is not None:
                if cnt >= num_batches:
                    print(f"Done till {num_batches} validation batches")
                    print("All specified batches done")
                    val_sanity_end = time.time()
                    print(f"Val sanity fit check passed in time {val_sanity_end-val_sanity_start}")
                    return True

    val_sanity_end = time.time()
    print("All specified batches done")
    print(f"Validation sanity check pased in time {val_sanity_end-val_sanity_start}")

    return True


def sanity_fit(model: nn.Module, train_loader, val_loader,
               device: str, num_batches: int = None,
               log_interval: int = 100, fp16: bool = False,):

    """
    Performs Sanity fit over train loader and valid loader.
    Use this to dummy check your fit function. It does not calculate metrics, timing, or does checkpointing.
    It iterates over both train_loader and val_loader for given batches.
    Note: - It does not to loss.backward().
    Args:
        model : A pytorch Faster RCNN Model.
        train_loader : Training loader.
        val_loader : Validation loader.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit sanity fit over certain batches.
                                 Useful is data is too big even for sanity check.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """

    sanity_train = train_sanity_fit(model, train_loader, device, num_batches, log_interval, fp16)

    sanity_val = val_sanity_fit(model, val_loader, device, num_batches, log_interval)

    return True
