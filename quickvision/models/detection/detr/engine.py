import torch
from torch.cuda import amp
from quickvision import utils
from tqdm import tqdm
import time
from collections import OrderedDict

__all__ = ["train_step", "val_step", "fit", "train_sanity_fit",
           "val_sanity_fit", "sanity_fit", ]


def train_step(model, train_loader, criterion, device, optimizer, scheduler=None,
               num_batches: int = None, log_interval: int = 100, scaler=None,):
    """
    Performs one step of training. Calculates loss, forward pass, computes gradient and returns metrics.
    Args:
        model : PyTorch Detr Model.
        train_loader : Train loader.
        device : "cuda" or "cpu"
        criterion : Detr Loss function to be optimized.
        optimizer : Torch optimizer to train.
        scheduler : Learning rate scheduler.
        num_batches : (optional) Integer To limit training to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        scaler: (optional)  Pass torch.cuda.amp.GradScaler() for fp16 precision Training.
    """

    model = model.to(device)
    criterion = criterion.to(device)
    start_train_step = time.time()
    model.train()
    last_idx = len(train_loader) - 1
    batch_time_m = utils.AverageMeter()
    criterion.train()
    cnt = 0
    batch_start = time.time()
    metrics = OrderedDict()

    total_loss = utils.AverageMeter()
    bbox_loss = utils.AverageMeter()
    giou_loss = utils.AverageMeter()
    labels_loss = utils.AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        images = list(image.to(device) for image in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                scaler.scale(loss).backward()
                # Step using scaler.step()
                scaler.step(optimizer)
                # Update for next iteration
                scaler.update()

        else:
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        cnt += 1
        total_loss.update(loss.item())
        bbox_loss.update(loss_dict["loss_bbox"].item())
        giou_loss.update(loss_dict["loss_giou"].item())
        labels_loss.update(loss_dict["loss_ce"].item())

        batch_time_m.update(time.time() - batch_start)
        batch_start = time.time()

        if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
            print("Batch Train Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  ".format(
                  batch_time=batch_time_m,))

        if num_batches is not None:
            if cnt >= num_batches:
                end_train_step = time.time()
                metrics["total_loss"] = total_loss.avg
                metrics["bbox_loss"] = bbox_loss.avg
                metrics["giou_loss"] = giou_loss.avg
                metrics["labels_loss"] = labels_loss.avg

                print(f"Done till {num_batches} train batches")
                print(f"Time taken for Training step = {end_train_step - start_train_step} sec")
                return metrics

    end_train_step = time.time()
    metrics["total_loss"] = total_loss.avg
    metrics["bbox_loss"] = bbox_loss.avg
    metrics["giou_loss"] = giou_loss.avg
    metrics["labels_loss"] = labels_loss.avg
    print(f"Time taken for Training step = {end_train_step - start_train_step} sec")
    return metrics


def val_step(model, val_loader, criterion, device,
             num_batches: int = None, log_interval: int = 100):
    """
    Performs one step of validation. Calculates loss, forward pass and returns metrics.
    Args:
        model : PyTorch Detr Model.
        val_loader : Validation loader.
        criterion : Detr Loss function to be optimized.
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
    criterion.eval()
    batch_start = time.time()
    metrics = OrderedDict()

    total_loss = utils.AverageMeter()
    bbox_loss = utils.AverageMeter()
    giou_loss = utils.AverageMeter()
    labels_loss = utils.AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            images = list(image.to(device) for image in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            cnt += 1
            total_loss.update(loss.item())
            bbox_loss.update(loss_dict["loss_bbox"].item())
            giou_loss.update(loss_dict["loss_giou"].item())
            labels_loss.update(loss_dict["loss_ce"].item())

            batch_time_m.update(time.time() - batch_start)
            batch_start = time.time()

            if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
                print("Batch Validation Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  ".format(
                      batch_time=batch_time_m,))

            if num_batches is not None:
                if cnt >= num_batches:
                    end_val_step = time.time()
                    metrics["total_loss"] = total_loss.avg
                    metrics["bbox_loss"] = bbox_loss.avg
                    metrics["giou_loss"] = giou_loss.avg
                    metrics["labels_loss"] = labels_loss.avg
                    print(f"Done till {num_batches} Validation batches")
                    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
                    return metrics

    end_val_step = time.time()
    metrics["total_loss"] = total_loss.avg
    metrics["bbox_loss"] = bbox_loss.avg
    metrics["giou_loss"] = giou_loss.avg
    metrics["labels_loss"] = labels_loss.avg
    print(f"Time taken for validation step = {end_val_step - start_val_step} sec")
    return metrics


def fit(model, epochs, train_loader, val_loader, criterion,
        device, optimizer, scheduler=None,
        num_batches: int = None, log_interval: int = 100,
        fp16: bool = False, ):

    """
    A fit function that performs training for certain number of epochs.
    Args:
        model : A pytorch Detr Model.
        epochs: Number of epochs to train.
        train_loader : Train loader.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        optimizer : PyTorch optimizer.
        scheduler : (optional) Learning Rate scheduler.
        early_stopper: (optional) A utils provied early stopper, based on validation loss.
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        fp16 : (optional) To use Mixed Precision Training using float16 dtype.
    """
    history = {}
    train_loss = []
    val_loss = []
    if fp16 is True:
        print("Training with Mixed precision fp16 scaler")
        scaler = amp.GradScaler()
    else:
        scaler = None

    for epoch in tqdm(range(epochs)):
        print()
        print(f"Training Epoch = {epoch}")
        train_metrics = train_step(model, train_loader, criterion,
                                   device, optimizer, scheduler, num_batches, log_interval,
                                   scaler=scaler, )

        train_loss.append(train_metrics["total_loss"])

        print(f"Validating Epoch = {epoch}")
        valid_metrics = val_step(model, val_loader, criterion, device, num_batches, log_interval)
        val_loss.append(valid_metrics["total_loss"])

    history = {"train": {"train_loss": train_loss},
               "val": {"val_loss": val_loss}}

    return history


def train_sanity_fit():
    pass


def val_sanity_fit():
    pass


def sanity_fit():
    pass
