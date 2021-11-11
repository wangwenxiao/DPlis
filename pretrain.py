import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np

from autodp import rdp_acct, rdp_bank
from model.model_dptrain import SmallCNN, SmallCNN_CIFAR, SmallCNN_CIFAR_SubEnsemble, Flatten
from model.model_dptrain_resnet import resnet18, resnet_Small
import model.model_face
from utils.data import get_data

import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--data', type=str, default='MNIST')

parser.add_argument('--runname', type=str, default="noname")
parser.add_argument('--save_epoch', type=int, default=99999)
parser.add_argument('--model', type=str, default='face_ResNet50')

parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_adj_period', type=int, default=99999)
parser.add_argument('--lr_adj_ratio', type=float, default=1.)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=100)

parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

if not os.path.isdir('./checkpoint/'):
    os.mkdir('./checkpoint/')
if not os.path.isdir('./log'):
    os.mkdir('./log')
logfile = os.path.join('./log', 'log_' + str(args.runname) + '.txt')
confgfile = os.path.join('./log', 'conf_' + str(args.runname) + '.txt')

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainset, testset, n_class = get_data(args.data)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4, drop_last = True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=4, shuffle=False)

device = 'cuda'

if args.model == 'face_ResNet50': #
    net = model.model_face.ResNet_50((112, 112), n_class)
    

net = net.to(device)
net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()   

opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)



for epoch in range(args.max_epoch):

    net.train()
    
    loss_train = 0.
    acc_train = 0.
    cnt = 0
    cnt_batch = 0


    _lr = float(args.lr * (args.lr_adj_ratio ** (epoch // args.lr_adj_period)))
    for param_group in opt.param_groups:
        param_group['lr'] = _lr

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        cnt += y.size(0)
        cnt_batch += 1
        
        output = net(x)
        loss = criterion(output, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss_train += torch.mean(loss).item()
        _, predicted = torch.max(output.data, 1)
        acc_train += predicted.eq(y.data).cpu().sum()
        if args.debug:
            print("[epoch %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, loss_train / cnt_batch, float(acc_train) / cnt))

    with open(logfile, 'a') as f:
        if args.debug:
            print("[epoch %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, loss_train / cnt_batch, float(acc_train) / cnt))
        f.write("[epoch %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, loss_train / cnt_batch, float(acc_train) / cnt))

    
    loss_test = 0
    acc_test = 0
    cnt_batch = 0
    cnt = 0

    net.eval()

    for x, y in test_loader:
        cnt_batch += 1
        cnt += x.size(0)
                
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            output = net(x)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, y)
            loss_test += torch.mean(loss).item()
            acc_test += predicted.eq(y.data).cpu().sum()
    with open(logfile, 'a') as f:
        if args.debug:
            print("[epoch %d][Loss_test: %.3f][Acc_test: %.4f]\n" % (epoch, loss_test / cnt_batch, float(acc_test) / cnt))
        f.write("[epoch %d][Loss_test: %.3f][Acc_test: %.4f]\n" % (epoch, loss_test / cnt_batch, float(acc_test) / cnt))

    
    if (epoch + 1) % args.save_epoch == 0:    
        ckp = {
            'net' : net,
            'loss_test' : loss_test / cnt_batch,
            'acc_test' : float(acc_test) / cnt, 
        }
        torch.save(ckp, './checkpoint/' + args.runname + str(epoch) + '.t7')

ckp = {
    'net' : net,
    'loss_test' : loss_test / cnt_batch,
    'acc_test' : float(acc_test) / cnt, 
}
torch.save(ckp, './checkpoint/' + args.runname + '.t7')


            



    




    














