import torch
import numpy as np
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SmallCNN(nn.Module):#model for mnist
    def __init__(self, num_of_classes = 10, in_channels = 1):
        super(SmallCNN, self).__init__()
        self.feature = torch.nn.Sequential(#1 * 28 * 28
            nn.Conv2d(in_channels, 16, 8, 2, padding=2), # 16 * 13 * 13
            nn.ReLU(),
            nn.MaxPool2d(2, 1), # 16 * 12 * 12
            nn.Conv2d(16, 32, 4, 2), #32 * 5 * 5
            nn.ReLU(),
            nn.MaxPool2d(2, 1), # 32 * 4 * 4
        )
        self.fc_layer = torch.nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, num_of_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return out


class SmallCNN_CIFAR(nn.Module):
    def __init__(self, num_of_classes = 10, in_channels = 3):
        super(SmallCNN_CIFAR, self).__init__()
        self.feature = torch.nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, 2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 4, 2), 
            nn.ReLU(),
            nn.MaxPool2d(2, 1), 
        )
        self.fc_layer = torch.nn.Sequential(
            nn.Linear(800, 32),
            nn.ReLU(),
            nn.Linear(32, num_of_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return out

class SmallCNN_CIFAR_TemperedSigmoid(nn.Module):
    def __init__(self, num_of_classes = 10, in_channels = 3):
        super(SmallCNN_CIFAR_TemperedSigmoid, self).__init__()
        self.feature = torch.nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, 2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 4, 2), 
            nn.Tanh(),
            nn.MaxPool2d(2, 1), 
        )
        self.fc_layer = torch.nn.Sequential(
            nn.Linear(800, 32),
            nn.Tanh(),
            nn.Linear(32, num_of_classes),
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return out

class SmallCNN_CIFAR_SubEnsemble(nn.Module):
    def __init__(self, num_of_classes = 10, in_channels = 3):
        super(SmallCNN_CIFAR_SubEnsemble, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, 2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(16, 32, 4, 2), 
            nn.ReLU(),
            nn.MaxPool2d(2, 1), 
        )
        self.fc_layer = torch.nn.Sequential(
            nn.Linear(800, 32),
            nn.ReLU(),
            nn.Linear(32, num_of_classes),
        )
        
        self.linear0 = nn.Linear(32 * 32 * 3, num_of_classes)
        self.linear1 = nn.Linear(3136, num_of_classes)
        self.linear2 = nn.Linear(800, num_of_classes)

    def forward(self, x):
        out0 = self.linear0(x.view(x.size(0), -1))
        
        conv1 = self.conv1(x)
        out1 = self.linear1(conv1.view(conv1.size(0), -1))

        conv2 = self.conv2(conv1)
        out2 = self.linear2(conv2.view(conv2.size(0), -1))
        
        
        out = self.fc_layer(conv2.view(conv2.size(0), -1))
        return out0 + out1 + out2 + out



