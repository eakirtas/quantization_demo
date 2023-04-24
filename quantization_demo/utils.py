import errno
import os

import numpy as np
import torch as T
import torchvision
import torchvision.transforms as transforms
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


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_DEFAULT_TRAIN_TRANFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

CIFAR10_DEFAULT_TEST_TRANFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_cifar10_dl(batch_size=32,
                   train_transforms=CIFAR10_DEFAULT_TRAIN_TRANFORM,
                   test_transforms=CIFAR10_DEFAULT_TEST_TRANFORM,
                   manual_seed=None,
                   num_workers=4,
                   download=True):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    trainset = torchvision.datasets.CIFAR10(root=ROOT_DIR,
                                            train=True,
                                            download=download,
                                            transform=train_transforms)

    trainloader = T.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          worker_init_fn=worker_init_fn)

    if test_transforms is None:
        test_transforms = train_transforms

    testset = torchvision.datasets.CIFAR10(root=ROOT_DIR,
                                           train=False,
                                           download=download,
                                           transform=test_transforms)

    testloader = T.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         worker_init_fn=worker_init_fn)

    return trainloader, testloader
