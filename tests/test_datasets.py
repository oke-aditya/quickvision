import unittest
import pandas as pd
import torchvision.transforms as T
from quickvision.datasets.classification import CSVDataset

df = pd.read_csv('tests/assets/csv_dataset.csv')
data_dir = 'tests/assets/'
tfms = T.Compose([
    T.ToTensor(),
])


class DatasetsTester(unittest.TestCase):
    def test_csv_dataset(self):
        flag = False
        complete_dataset = CSVDataset(df, data_dir, 'Image', 'Label', tfms, 'png')
        complete_dataset[0]
        flag = True
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
