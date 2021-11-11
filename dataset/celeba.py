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

class CELEBA(data.Dataset):
    # mode in ["pretrain_train", "pretrain_test", "private_train", "private_test", "all_train", "all_test"]
    # private train&test: 1000 classes such that 'num-per-class=30', in which 25 within each class will become train and the other 5 as test
    # pretrain train&test: all remained classes, with 1 from each class reserved for test, and the others for train(44 classes with only 1 sample are ignored)
    # all train&test: all classes, with 1 from each class reserved for test, and the others for train(44 classes with only 1 sample are ignored)

    def __init__(self, root='~/data/', mode='pretrain_train', transform=None, preload = True):

        root = os.path.expanduser(root)
    
        self.lbl_path = os.path.join(root, 'identity_CelebA.txt')
        self.root =  os.path.join(root, 'img_align_celeba/')

        self.preload = preload

        self.imgs = []
        self.loader = default_loader
        self.transform = transform
        self.private_num_per_class = 30
        self.private_train_per_class = 25
        self.private_n_class = 1000
        self.mode = mode
        
        with open(self.lbl_path, 'r') as f:
            for line in f:
                img, lbl = line.split()
                self.imgs.append((self.root + img, int(lbl)))

        cnt = {}
        for img, lbl in self.imgs:
            if lbl not in cnt:
                cnt[lbl] = 1
            else:
                cnt[lbl] += 1

#        cnt_cnt = [0 for i in range(50)]
#        for key in cnt.keys():
#            cnt_cnt[cnt[key]] += 1
#        print (cnt_cnt)

        self.n_class = 0
        _imgs = self.imgs
        self.imgs = []
        self.mp = {}
        final_cnt = {}


        #find all private classes
        for img, lbl in _imgs:
            if cnt[lbl] == self.private_num_per_class:
                if lbl not in self.mp:
                    if self.n_class == self.private_n_class:
                        continue
                    self.mp[lbl] = self.n_class
                    final_cnt[self.n_class] = 0
                    self.n_class += 1
                self.imgs.append((img, self.mp[lbl]))
                final_cnt[self.mp[lbl]] += 1

        if mode.split('_')[0] == 'pretrain':
            _mp = self.mp
            self.mp = {}
            self.n_class = 0
            self.imgs = []
            final_cnt = {}
        
            for img, lbl in _imgs:
                if (lbl not in _mp) and cnt[lbl] > 1: #not in a private class & contain at least 2 samples
                    if lbl not in self.mp:
                        self.mp[lbl] = self.n_class
                        final_cnt[self.n_class] = 0
                        self.n_class += 1
                    self.imgs.append((img, self.mp[lbl]))
                    final_cnt[self.mp[lbl]] += 1
        elif mode.split('_')[0] == 'all':
            _mp = self.mp
            self.mp = {}
            self.n_class = 0
            self.imgs = []
            final_cnt = {}
        
            for img, lbl in _imgs:
                if cnt[lbl] > 1: #contain at least 2 samples
                    if lbl not in self.mp:
                        self.mp[lbl] = self.n_class
                        final_cnt[self.n_class] = 0
                        self.n_class += 1
                    self.imgs.append((img, self.mp[lbl]))
                    final_cnt[self.mp[lbl]] += 1



        print (len(self.imgs))
        print (self.n_class)

        if mode.split('_')[0] == 'private':
            test_per_class = self.private_num_per_class - self.private_train_per_class
        elif mode.split('_')[0] == 'pretrain':
            test_per_class = 1
        elif mode.split('_')[0] == 'all':
            test_per_class = 1
        else:
            raise Exception("check the mode of using celeba!!")

        _imgs = self.imgs
        self.imgs = []
        for img, lbl in _imgs:
            final_cnt[lbl] -= 1
            if (mode.split('_')[1] == 'test' and final_cnt[lbl] < test_per_class) or (mode.split('_')[1] == 'train' and final_cnt[lbl] >= test_per_class):
                self.imgs.append((img, lbl))

        if self.preload:
            _imgs = self.imgs
            self.imgs = []
            for path, target in _imgs:
                self.imgs.append((self.loader(path).crop((0, 40, 178, 218)), target))

        print (len(self.imgs))
                        

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
            sample = self.loader(path).crop((0, 40, 178, 218))
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    A = CELEBA()
