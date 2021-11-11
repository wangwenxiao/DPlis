import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from dataset import celeba,pubfig83

def get_data(datatype='MNIST'):
    root = os.path.expanduser('~/data')

    if datatype == 'MNIST':
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                ])
        transform_test = transforms.Compose([
                 transforms.ToTensor(),
                ])

        trainset = torchvision.datasets.MNIST(root=root, 
                    train=True, download=False,transform=transform_train)
        testset = torchvision.datasets.MNIST(root=root,
                    train=False, download=False,transform=transform_test)
        n_class = 10


    elif datatype.split('+')[0] == 'CIFAR10':
        if len(datatype.split('+')) > 1:
            if datatype.split('+')[1] == 'cropflip':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
                
        trainset = torchvision.datasets.CIFAR10(root=root, 
                    train=True, download=True,transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root,
                    train=False, download=True,transform=transform_test)
        n_class = 10


    elif datatype.split('+')[0] == 'celeba':
        mode = datatype.split('+')[1] #mode in ['pretrain', 'private']
        
        
        transform_train = transforms.Compose([
            transforms.CenterCrop((128, 128)),
            transforms.RandomCrop((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = celeba.CELEBA(root = root, mode = mode + '_train', transform = transform_train)
        testset = celeba.CELEBA(root = root, mode = mode + '_test', transform = transform_test)

        n_class = trainset.n_class

    elif datatype == 'pubfig83-aligned':
        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomCrop((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = pubfig83.PUBFIG83(root = root, train = True, transform = transform_train)
        testset = pubfig83.PUBFIG83(root = root, train = False, transform = transform_test)

        n_class = trainset.n_class

    elif datatype == 'resizedSTL10':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        trainset = torchvision.datasets.STL10(root=root, 
                    split='train', download=True,transform=transform_train)
        testset = torchvision.datasets.STL10(root=root,
                    split='test', download=True,transform=transform_test)
        n_class = 10
        

    elif datatype == 'resizedCIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        trainset = torchvision.datasets.CIFAR10(root=root, 
                    train=True, download=True,transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root,
                    train=False, download=True,transform=transform_test)
        n_class = 10
    elif datatype == 'resizedCIFAR100':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        trainset = torchvision.datasets.CIFAR100(root=root, 
                    train=True, download=True,transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root,
                    train=False, download=True,transform=transform_test)
        n_class = 100
        
    
    return trainset, testset, n_class

class RandomBatchSampler(object):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_data = len(self.data_source)
        self.mask = torch.ones(self.num_data)
        self.batch_per_epoch = self.num_data // self.batch_size
        self.rate = float(self.batch_size) / self.num_data

    def __iter__(self):
        for i in range(self.batch_per_epoch):
            actual_batch_size = (torch.rand(self.num_data) < self.rate).sum()
            yield torch.multinomial(self.mask, num_samples = actual_batch_size, replacement=False)

    def __len__(self):
        return self.batch_per_epoch


class DataSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, id_min, id_max, transform=None):
        self.dataset = dataset
        self.id_min = id_min
        self.id_max = id_max
        self.transform = transform

        assert(id_min >= 0)
        assert(id_min < id_max)
        assert(len(self.dataset) >= id_max)

    def __getitem__(self, index):
        
        img, label = self.dataset[self.id_min + index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.id_max - self.id_min





