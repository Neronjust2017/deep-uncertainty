import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from base import BaseDataLoader, BaseDataLoader_2
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BostonHousingDataLoader(BaseDataLoader):
    """
    BostonHousing data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        from sklearn.datasets import load_boston
        boston = load_boston()
        X = boston.data
        Y = boston.target
        Y = Y.reshape([Y.shape[0],1])
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        dataset= TensorDataset(X, Y)
        self.data_dir = data_dir
        super().__init__(dataset, batch_size, shuffle, validation_split, test_split, num_workers, normalization=True)

class MonthlyCarSalesDataLoader(BaseDataLoader_2):
    """
    MonthlyCarSales data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, n_input=36, n_val=12, n_test=12, num_workers=1, training=True):
        self.data_dir = data_dir
        series = pd.read_csv(data_dir, header=0, index_col=0)
        data = series.values
        data = self.series_to_supervised(data, n_in=n_input)
        X = data[:, :-1]
        Y = data[:, -1]

        idx_full = np.arange(len(X))
        test_idx = idx_full[-n_test:]
        valid_idx = idx_full[-(n_test + n_val):-n_test]
        train_idx = idx_full[:-(n_test + n_val)]

        X = X.reshape([X.shape[0], 1, X.shape[1]])
        Y = Y.reshape([Y.shape[0], 1])

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

        dataset = TensorDataset(X, Y)

        super().__init__(dataset, batch_size, shuffle, train_idx, valid_idx, test_idx, num_workers, normalization=True)

    def series_to_supervised(self, data, n_in=1, n_out=1):
        df = pd.DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg.values
