# Add code for Mapping using dataframe containaing id and target.
# Port from pytorch_cnn_trainer
# https://github.com/oke-aditya/pytorch_cnn_trainer

import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
import os
import torch
from PIL import Image

__all__ = ["create_folder_dataset", "CSVDataset", ]


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


class CSVDataset(Dataset):
    """
    Creates Torchvision Dataset From CSV File.
    Args:
        df: DataFrame with a column 2 columns image_id and target
        data_dir: Directory from where data is to be read.
        target: target column name
        transform: Trasforms to apply while creating Dataset.
    """

    def __init__(self, df, data_dir, target, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.image_id[idx]
        label = self.df.target[idx]

        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)

        return image, label
