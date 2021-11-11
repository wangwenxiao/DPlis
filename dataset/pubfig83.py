from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch.utils.data as data
from torchvision.datasets import ImageFolder

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def is_valid_file(path):
    return not path.split('/')[-1].startswith('.')

class PUBFIG83(data.Dataset):
    def __init__(self, root='~/data/', train=True, test_perclass=50, transform=None, preload = True):
        root = os.path.expanduser(root)
        root = os.path.join(root, 'pubfig83-aligned')
        self.root = root
        self.imgs_all = ImageFolder(root).imgs
        
        self.n_class = 83
        self.train = train
        self.test_perclass = test_perclass
        self.transform = transform
        self.loader = default_loader
        self.preload = preload

        cnt = [0 for i in range(self.n_class)]
        for img, lbl in self.imgs_all:
            cnt[lbl] += 1
        
        self.imgs = []
        for img, lbl in self.imgs_all:
            cnt[lbl] -= 1
            if (train and cnt[lbl] >= test_perclass ) or ((not train) and cnt[lbl] < test_perclass):
                self.imgs.append((img, lbl))
        print (len(self.imgs))

        if self.preload:
            _imgs = self.imgs
            self.imgs = []
            for path, target in _imgs:
                self.imgs.append((self.loader(path), target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        if self.preload:
            sample = path
        else:
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.imgs)
