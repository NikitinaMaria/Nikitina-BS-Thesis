import typing

import torch
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10


class CIFAR10Pair(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool,
                 transform: typing.Callable,
                 download: bool = True,
                 noise_frac: typing.Optional[float] = None):
        super().__init__(root, train, transform, None, download)
        self.targets = np.array(self.targets)
        self.labels = self.targets
        self.noise_frac = noise_frac
        assert self.transform is not None, "Empty transform"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        pos_1 = self.transform(img)

        if self.noise_frac is not None and np.random.rand() < self.noise_frac:
            index = np.where(self.targets != target)[0]
            assert len(index) > 1000, "Bad query"
            img_rand = self.data[np.random.choice(index)]
            img_rand = Image.fromarray(img_rand)
            pos_2 = self.transform(img_rand)
        else:
            pos_2 = self.transform(img)
        return pos_1, pos_2, target

    def extra_repr(self):
        return "Noise frac: {}".format(self.noise_frac)


def get_dataset(name: str, root: str, split: str, transform=None, noise_frac=None) -> torch.utils.data.Dataset:
    if name == "CIFAR10":
        return CIFAR10Pair(root, train="train" in split, transform=transform, noise_frac=noise_frac, )
    raise Exception("Unknown dataset {}".format(name))
