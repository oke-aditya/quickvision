
# Training Object Detection Models with Quickvision !

# Defining the Dataset

# The dataset should inherit from the standard
# `torch.utils.data.Dataset` class, and implement `__len__` and `__getitem__`.

# The only specificity that we require is that the dataset `__getitem__` should return:

# * image: a PIL Image of size (H, W)
# * target: a dict containing the following fields
# * `boxes` (`FloatTensor[N, 4]`): the coordinates of the
#   `N` bounding boxes in `[x0, y0, x1, y1]` format, ranging from `0` to `W` and `0` to `H`
# * `labels` (`Int64Tensor[N]`): the label for each bounding box

# Writing a custom dataset for Penn-Fudan

# Let's write a dataset for the Penn-Fudan dataset.

# First, let's download and extract the data, present in
# a zip file at https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

# download the Penn-Fudan dataset
# $ wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# $ extract it in the current folder
# $ unzip PennFudanPed.zip

# Let's have a look at the dataset and how it is layed down.

# The data is structured as follows

# PennFudanPed/
#   PedMasks/
#     FudanPed00001_mask.png
#     FudanPed00002_mask.png
#     FudanPed00003_mask.png
#     FudanPed00004_mask.png
#     ...
#   PNGImages/
#     FudanPed00001.png
#     FudanPed00002.png
#     FudanPed00003.png
#     FudanPed00004.png

import os
import numpy as np
import torch
import torch.utils.data
from pprint import pprint
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from quickvision.models.detection import faster_rcnn
from quickvision.models.detection import retinanet
from torch.cuda import amp


if __name__ == "__main__":

    # Here is one example of an image in the dataset, with its corresponding instance segmentation mask
    Image.open('PennFudanPed/PNGImages/FudanPed00001.png')

    # So each image has a corresponding segmentation mask, where
    # each color correspond to a different instance.
    # Let's write a `torch.utils.data.Dataset` class for this dataset.

    class PennFudanDataset(torch.utils.data.Dataset):
        def __init__(self, root):
            self.root = root
            # load all image files, sorting them to
            # ensure that they are aligned
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        def __getitem__(self, idx):
            # load images ad masks
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            # note that we haven't converted the mask to RGB,
            # because each color corresponds to a different instance
            # with 0 being background
            mask = Image.open(mask_path)

            mask = np.array(mask)
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            # target["masks"] = masks
            # target["image_id"] = image_id
            target["area"] = area
            # target["iscrowd"] = iscrowd

            img = T.ToTensor()(img)

            return img, target

        def __len__(self):
            return len(self.imgs)

    # That's all for the dataset. Let's see how the outputs are structured for this dataset

    # So we can see that by default, the dataset returns a `PIL.Image` and a dictionary
    # containing several fields, including `boxes`, `labels`.

    # Putting everything together

    # We now have the dataset class, the models and the data transforms. Let's instantiate them

    def collate_fn(batch):
        return tuple(zip(*batch))

    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed')
    dataset_test = PennFudanDataset('PennFudanPed')

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                               num_workers=4, collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False,
                                              num_workers=4, collate_fn=collate_fn)

    # Now let's instantiate the model and the optimizer

    # And now let's train the model for 10 epochs, evaluating at the end of every epoch.

    # Quickvision to Train Detection Models !

    # Training Faster RCNN

    backbone = faster_rcnn.create_fastercnn_backbone("resnet101", fpn=True, pretrained="imagenet",
                                                     trainable_backbone_layers=3)

    # our dataset has two classes only - background and person
    num_classes = 2
    model = faster_rcnn.create_vision_fastercnn(num_classes=num_classes, backbone=backbone)

    # - Quickvision supports Mixed Precision training as well !

    # Let's use Mixed Precision training

    scaler = amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 3

    for epoch in tqdm(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        train_metrics = faster_rcnn.train_step(model, train_loader,
                                               device="cuda", optimizer=optimizer,
                                               scheduler=lr_scheduler, log_interval=100, scaler=scaler)
        val_metrics = faster_rcnn.val_step(model, test_loader, device="cuda")
        print("Training Metrics: ")
        pprint(train_metrics)
        print("Validation metrics")
        pprint(val_metrics)

    # Training Retina Net

    backbone = retinanet.create_retinanet_backbone("resnet50", fpn=True,
                                                   pretrained="imagenet", trainable_backbone_layers=3)

    # our dataset has two classes only - background and person
    num_classes = 2
    model = retinanet.create_vision_retinanet(num_classes=num_classes, backbone=backbone)

    # Let's use Mixed Precision training

    scaler = amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 3

    for epoch in tqdm(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        train_metrics = retinanet.train_step(model, train_loader, device="cuda",
                                             optimizer=optimizer, scheduler=lr_scheduler,
                                             log_interval=100, scaler=scaler)

        val_metrics = retinanet.val_step(model, test_loader, device="cuda")
        print("Training Metrics: ")
        pprint(train_metrics)
        print("Validation metrics")
        pprint(val_metrics)
