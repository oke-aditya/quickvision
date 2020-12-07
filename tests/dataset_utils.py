import torchvision
import torch
import torchvision.transforms as T
from torchvision import ops
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

__all__ = ["create_cifar10_dataset", "create_loaders", "DummyDetectionDataset"]


class DummyDetectionDataset(Dataset):
    """
    Generate a dummy dataset for detection
    Example::
        >>> ds = DummyDetectionDataset()
        >>> dl = DataLoader(ds, batch_size=7)
    """
    def __init__(self, img_shape: tuple = (3, 256, 256), num_boxes: int = 1,
                 num_classes: int = 2, class_start: int = 0,
                 num_samples: int = 10000, box_fmt: str = "xyxy",
                 normalize: bool = False):
        """
        Args:
            *shapes: list of shapes
            img_shape: Tuple of (channels, height, width)
            num_boxes: Number of boxes per images
            num_classes: Number of classes for image.
            num_samples: how many samples to use in this dataset.
            box_fmt: Format of Bounding boxes, supported : "xyxy", "xywh", "cxcywh"
        """
        super().__init__()
        self.img_shape = img_shape
        self.num_samples = num_samples
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.box_fmt = box_fmt
        self.class_start = class_start
        self.class_end = self.class_start + self.num_classes
        self.normalize = normalize

    def __len__(self):
        return self.num_samples

    def _random_bbox(self):
        c, h, w = self.img_shape
        xs = torch.randint(w, (2,), dtype=torch.float32)
        ys = torch.randint(h, (2,), dtype=torch.float32)
        if self.normalize:
            xs /= self.img_shape[0]  # divide by the width
            ys /= self.img_shape[1]  # divide by the height
        # A small hacky fix to avoid degenerate boxes.
        return [min(xs), min(ys), max(xs) + 0.1, max(ys) + 0.1]

    def __getitem__(self, idx: int):
        img = torch.rand(self.img_shape)
        boxes = torch.tensor([self._random_bbox() for _ in range(self.num_boxes)], dtype=torch.float32)
        boxes = ops.clip_boxes_to_image(boxes, (self.img_shape[1], self.img_shape[2]))
        # No problems if we pass same in_fmt and out_fmt, it is covered by box_convert
        converted_boxes = ops.box_convert(boxes, in_fmt="xyxy", out_fmt=self.box_fmt)
        labels = torch.randint(self.class_start, self.class_end, (self.num_boxes,), dtype=torch.long)
        return img, {"boxes": converted_boxes, "labels": labels}


def create_cifar10_dataset():
    """
    Creates CIFAR10 train dataset and a test dataset.
    """

    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    train_set = datasets.CIFAR10("./data", download=True, train=True, transform=train_transforms)
    valid_set = datasets.CIFAR10("./data", download=True, train=False, transform=valid_transforms)

    return train_set, valid_set


def create_loaders(train_dataset, valid_dataset, train_batch_size=32, valid_batch_size=32, num_workers=4):

    """
    Creates train loader and test loader from train and test datasets
    Args:
        train_dataset: Torchvision train dataset.
        valid_dataset: Torchvision valid dataset.
        train_batch_size (int) : Default 32, Training Batch size
        valid_batch_size (int) : Default 32, Validation Batch size
        num_workers (int) : Defualt 1, Number of workers for training and validation.
    """

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True,
                              num_workers=num_workers)

    valid_loader = DataLoader(valid_dataset, valid_batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, valid_loader
