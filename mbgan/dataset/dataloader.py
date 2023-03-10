import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MBDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.s16_csv = self.config.data.s16_csv
        self.wgs_csv = self.config.data.wgs_csv

        self.df_16s = None
        self.df_wgs = None

        self.tensor_16s = None
        self.tensor_wgs = None

        self.preprocess_data()

    def preprocess_data(self):
        self.df_16s = pd.read_csv(self.s16_csv).drop(['ID'], axis=1)
        self.df_wgs = pd.read_csv(self.wgs_csv).drop(['ID'], axis=1)

        self.tensor_16s = torch.tensor(self.df_16s.values).float()
        self.tensor_wgs = torch.tensor(self.df_wgs.values).float()

    def __getitem__(self, index):
        return self.tensor_16s[index], self.tensor_wgs[index]

    def __len__(self):
        return min(len(self.df_16s), len(self.df_wgs))


class MBDataloader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, config, shuffle=True):
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = self.config.trainer.batch_size
        self.collate_fn = None

        self.batch_idx = 0
        self.n_samples = len(dataset)

        if self.collate_fn is None:
            super().__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                             num_workers=4, drop_last=True)
        else:
            super().__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                             collate_fn=self.collate_fn, num_workers=4, drop_last=True)
