from pathlib import Path

import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader_Stage2(dataset):
    def __init__(self):
        super().__init__('MNIST', 'MNIST digit classification')
        self.data_folder = Path(__file__).resolve().parents[2] / 'data' / 'stage_2_data'

    def load(self):
        print('loading Stage 2 MNIST data...')

        train_data = np.loadtxt(self.data_folder / 'train.csv', delimiter=',', dtype=np.float32)
        test_data = np.loadtxt(self.data_folder / 'test.csv', delimiter=',', dtype=np.float32)

        train_y = train_data[:, 0].astype(np.int64)
        train_x = train_data[:, 1:] / 255.0
        test_y = test_data[:, 0].astype(np.int64)
        test_x = test_data[:, 1:] / 255.0

        return {
            'train': {'X': train_x, 'y': train_y},
            'test': {'X': test_x, 'y': test_y},
        }
