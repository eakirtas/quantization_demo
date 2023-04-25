import errno
import os

import numpy as np
from data_utils.utils.files import ifnot_create
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor


def ifnot_create(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return path


ROOT_DIR = ifnot_create('datasets/')

DEFAULT_TRANSFORMS = Compose([
    ToTensor(),
])


def get_mnist(batch_size,
              transforms=DEFAULT_TRANSFORMS,
              manual_seed=None,
              generator=None,
              num_workers=4):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    train_mnist = MNIST(ROOT_DIR + "train",
                        transform=transforms,
                        train=True,
                        download=True)
    test_mnist = MNIST(ROOT_DIR + "test",
                       transform=transforms,
                       train=False,
                       download=True)

    train_mnist = DataLoader(train_mnist,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             worker_init_fn=worker_init_fn,
                             generator=generator)

    test_mnist = DataLoader(test_mnist,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=worker_init_fn,
                            generator=generator)

    # plot_mnist(dataloader_dict['train']['dataloader'], plot_path, tune_path)
    return train_mnist, test_mnist
