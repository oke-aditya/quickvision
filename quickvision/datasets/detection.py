import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

__all__ = ["frcnn_dataset"]


class frcnn_dataset(Dataset):
    """
    df: DataFrame having columns "image_id,
    """
    def __init__(self, df, image_dir, target, transforms=None, train=True):
        super().__init__()
        self.image_ids = df["image_id"].unique()
        self.image_dir = image_dir
        self.transforms = transforms
        self.df = df
        self.train = train
        self.target = target

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_src = os.path.join(self.image_dir, str(image_id))
