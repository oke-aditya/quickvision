import unittest
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as T
from quickvision.datasets.classification import CSVSingleLabelDataset
from quickvision.models.classification import cnn

df = pd.read_csv("tests/assets/csv_dataset.csv")
data_dir = "tests/assets/"
tfms = T.Compose([T.ToTensor(),])


class CSVSingleLabelDatasetTester(unittest.TestCase):
    def test_csv_single_label_dataset(self):
        complete_dataset = CSVSingleLabelDataset(
            df, data_dir, "Image", "Label", tfms, "png"
        )
        self.assertTrue(complete_dataset[0])

        train_loader = torch.utils.data.DataLoader(complete_dataset, num_workers=1)
        model = cnn.create_cnn("resnet18", 2, pretrained=None)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = nn.CrossEntropyLoss()
        res = cnn.train_sanity_fit(model, train_loader, loss, "cpu", num_batches=1)
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
