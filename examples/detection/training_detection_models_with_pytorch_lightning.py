# Training Detection Models with Quickvision and  PyTorch Lightning

# This tutorial follows the dataset creation from torchvision tutorial as well as from Quickvision PyTorch tutorial.

# Feel free to skip that part.

# Writing a custom dataset for Penn-Fudan

# Let's write a dataset for the Penn-Fudan dataset.

# First, let's download and extract the data, present in a
# zip file at https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

# download the Penn-Fudan dataset
# wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# extract it in the current folder
# unzip PennFudanPed.zip

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


# Here is one example of an image in the dataset, with its corresponding instance segmentation mask

# from PIL import Image
# Image.open('PennFudanPed/PNGImages/FudanPed00001.png')

import os
import numpy as np
import torch
import torch.utils.data
from pprint import pprint
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import pytorch_lightning as pl
from quickvision.models.detection.faster_rcnn import lit_frcnn
from quickvision.models.detection.retinanet import lit_retinanet


if __name__ == "__main__":

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

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                              shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Training using PyTorch Lightning !

    # !pip install -q quickvision

    model = lit_frcnn(learning_rate=1e-3, num_classes=2, backbone="resnet50",
                      fpn=True, pretrained_backbone="imagenet")

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(model, train_loader, test_loader)

    # Custom Training with Lightning !

    # - To write your own Training logic, metrics, logging. Subclass the `lit_frcnn` and write your own logic !

    class CustomTraining(lit_frcnn):
        def training_step(self, batch, batch_idx):
            images, targets = batch
            targets = [{k: v for k, v in t.items()} for t in targets]

            # fasterrcnn takes both images and targets for training, returns
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            return {"loss": loss, "log": loss_dict}

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    custom_model = CustomTraining(learning_rate=1e-3, num_classes=2,
                                  backbone="resnet50", fpn=True, pretrained_backbone="imagenet")

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(custom_model, train_loader, test_loader)

    # Training RetinaNet with Lightning

    model = lit_retinanet(learning_rate=1e-3, num_classes=2,
                          backbone="resnet50", fpn=True, pretrained_backbone="imagenet")

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(model, train_loader, test_loader)

    # Custom Training with Lightning

    class CustomTraining(lit_retinanet):
        def training_step(self, batch, batch_idx):
            images, targets = batch
            targets = [{k: v for k, v in t.items()} for t in targets]

            # fasterrcnn takes both images and targets for training, returns
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            return {"loss": loss, "log": loss_dict}

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    custom_retina_model = CustomTraining(learning_rate=1e-3, num_classes=2,
                                         backbone="resnet50", fpn=True, pretrained_backbone="imagenet")

    trainer = pl.Trainer(max_epochs=2, gpus=1)
    trainer.fit(custom_retina_model, train_loader, test_loader)
