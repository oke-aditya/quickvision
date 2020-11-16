# Add code for Mapping using dataframe containaing id and target.
# Port from pytorch_cnn_trainer
# https://github.com/oke-aditya/pytorch_cnn_trainer

import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
import os
import torch
from PIL import Image

__all__ = ["create_folder_dataset", "CSVSingleLabelDataset"]


def create_folder_dataset(root_dir, transforms, split: float = 0.8, **kwargs):
    """
    Creates Train and Validation Dataset from a Root folder
    Arrange dataset as follows: -
    root/class_a/image01.png
    root/class_b/image01.png

    Creates train and validation dataset from this root dir.
    This applies same transforms to both train and validation

    Args:
    root_dir : Root directory of the dataset to read from
    transforms: Transforms to be applied to train and validation datasets.
    split: Float number denoting percentage of train items

    """
    complete_dataset = datasets.ImageFolder(root_dir, transform=transforms)
    train_split = len(complete_dataset) * split
    valid_split = len(complete_dataset) * (1 - split)

    train_set, valid_set = torch.utils.data.random_split(complete_dataset, [train_split, valid_split])
    return train_set, valid_set


class CSVSingleLabelDataset(Dataset):
    """
    Creates Torchvision Dataset From CSV File.
    Args:
        df: DataFrame with 2 columns ``image_id`` and ``target``.
        data_dir: Directory from where data is to be read.
        image_id: Column name which has IDs of the images.
        target: target column name.
        transform: Trasforms to apply while creating Dataset.
        img_type: Type of the image like `png` or `jpg` etc.
    """

    def __init__(self, df, data_dir, image_id, target, transform, img_type):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.image_id = image_id
        self.target = target
        self.transform = transform
        self.img_type = img_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df[self.image_id][idx]
        label = self.df[self.target][idx]

        img_path = os.path.join(self.data_dir, str(img_name) + f'.{self.img_type}')
        image = Image.open(img_path)
        image = self.transform(image)

        return image, label
