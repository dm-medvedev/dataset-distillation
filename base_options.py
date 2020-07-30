import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class PlaneData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_plane_dataset(type_='moons', n_samples=1500, test_size=0.33, random_state=8):
    if type_ == 'blobs':
        X, y = dataset.smake_blobs(n_samples=n_samples, random_state=random_state)
    elif type_ == 'moons':
        X, y = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    elif type_ == 'circles':
        X, y = dataset.smake_circles(n_samples=n_samples, factor=5, noise=0.05, random_state=random_state)
    else:
        raise ValueError('Unsupported dataset: %s' % type_)
    X = np.array(X, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_dataset = PlaneData(X_train, y_train)
    test_dataset = PlaneData(X_test, y_test)
    return train_dataset, test_dataset


class State(dict):
    def __init__(self, *args, **kwargs):
        super(State, self).__init__(*args, **kwargs)
        self.__dict__ = self

RES_FOLDER = './Results'

BASE_OPTIONS = {'distill_steps': 1,
                'distill_epochs': 1,
                'num_classes': 2,
                'distilled_images_per_class_per_step': 4,
                'distill_lr': 0.02,
                'lr': 0.01,
                'device': torch.device('cuda: 0'),
                'input_size': 2,
                'distributed': False,
                'decay_epochs': 1,
                'decay_factor': 1.,
                'num_workers': 8,
                'batch_size': 64,
                'checkpoint_interval': 100_000,
                'epochs': 10,
                'output_flag': False,
                'test_nets_type': 'unknown_init',
                'arch': None,
                'init': 'xavier',
                'init_param': 1,
                'mode': 'distill_basic',
                'test_niter': 1,
                'train_nets_type': 'unknown_init',
                'log_interval': 100_000,
                'L2_coef': 1e-1,
                'local_n_nets': None,
                }

np.random.seed(42)
TRAIN_DATASET, TEST_DATASET = get_plane_dataset('moons')
BASE_OPTIONS['train_loader'] = DataLoader(TRAIN_DATASET,
                                          batch_size=BASE_OPTIONS['batch_size'],
                                          num_workers=BASE_OPTIONS['num_workers'],
                                          pin_memory=True, shuffle=True)
BASE_OPTIONS['test_loader'] = DataLoader(TEST_DATASET,
                                         batch_size=BASE_OPTIONS['batch_size'],
                                         num_workers=BASE_OPTIONS['num_workers'],
                                         pin_memory=True, shuffle=False)
