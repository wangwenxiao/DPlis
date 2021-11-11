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
from model.model_dptrain import SmallCNN_CIFAR_TemperedSigmoid
from model.model_dptrain_resnet import resnet18, resnet_Small
import model.model_face
from utils.data import get_data, RandomBatchSampler

import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--data', type=str, default='MNIST')
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--noise_multiplier', type=float, default=1)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--runname', type=str, default="noname")
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--g_clip', type=float, default=1.)
parser.add_argument('--mode', type=str, default='naive', help='mode in [naive, multi-batch, smooth_loss')
parser.add_argument('--T_multi', type=int, default=0, help='number of batches')
parser.add_argument('--save_epoch', type=int, default=99999)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no_privacy', action='store_true')
parser.add_argument('--model', type=str, default='SmallCNN')
parser.add_argument('--frequent_ckp_last10epoch', action='store_true')
parser.add_argument('--frequent_ckp_last20epoch', action='store_true')

parser.add_argument('--adaptive_clipping', action='store_true')
parser.add_argument('--init_clip', type=float, default=500)
parser.add_argument('--adaptive_period', type=int, default=50)

parser.add_argument('--adaptive_lr', action='store_true')
parser.add_argument('--adaptive_lr_T', type=float, default=1.0)

parser.add_argument('--lr_adj_period', type=int, default=99999)
parser.add_argument('--lr_adj_ratio', type=float, default=1.)

parser.add_argument('--ExpWeightAvg', type=float, default=0)


parser.add_argument('--grad_zero_out', type=float, default=0)

parser.add_argument('--L1_grad_coef', type=float, default=0)

parser.add_argument('--smooth_loss_samples', type=int, default=1)
parser.add_argument('--smooth_loss_radius', type=float, default=1.0)

parser.add_argument('--toy_example_for_smooth_radius', action='store_true')
parser.add_argument('--toy_example_for_smooth_radius_batch_coef', type=float, default=1.0)

parser.add_argument('--pretrain_feat', type=str, default='')
parser.add_argument('--clip_no_privacy', action='store_true')

parser.add_argument('--efficientnet_pretrained', type=str, default='')

parser.add_argument('--n_worker', type=int, default=4)




args = parser.parse_args()

frequent_ckp_cnt = 0

if not os.path.isdir('./checkpoint/'):
    os.mkdir('./checkpoint/')
if not os.path.isdir('./imgs'):
    os.mkdir('./imgs')
if not os.path.isdir('./imgs/'+args.runname):
    os.mkdir('./imgs/'+args.runname)
if not os.path.isdir('./log'):
    os.mkdir('./log')
logfile = os.path.join('./log', 'log_' + str(args.runname) + '.txt')

confgfile = os.path.join('./log', 'conf_' + str(args.runname) + '.txt')

# save configuration parameters
with open(confgfile, 'w') as f:
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))

trainset, testset, n_class = get_data(args.data)

#privacy accoutant
N = trainset.__len__()
acct = rdp_acct.anaRDPacct()
func = lambda x: rdp_bank.RDP_gaussian({'sigma': args.noise_multiplier}, x)

batch_sampler = RandomBatchSampler(trainset, args.batch_size)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, batch_sampler = batch_sampler, num_workers=args.n_worker)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.n_worker, shuffle=False)

device = 'cuda'

if args.model == 'SmallCNN':
    if args.data == 'MNIST':
        net = SmallCNN(n_class, args.in_channels)
    else:
        net = SmallCNN_CIFAR(n_class, args.in_channels)
elif args.model == 'SmallCNN_TemperedSigmoid':
    net = SmallCNN_CIFAR_TemperedSigmoid(n_class, args.in_channels)
elif args.model == 'SmallCNN_SubEnsemble':
    net = SmallCNN_CIFAR_SubEnsemble(n_class, args.in_channels)
elif args.model.split('+')[0] == 'ResNet18':
    if args.model.split('+')[1] == 'InstanceNorm':
        net = resnet18(n_class = n_class, norm_layer = torch.nn.InstanceNorm2d)
elif args.model.split('+')[0] == 'ResNet_Small':
    if args.model.split('+')[1] == 'InstanceNorm':
        net = resnet_Small(n_class = n_class, norm_layer = torch.nn.InstanceNorm2d)

elif args.model == 'LinearMNIST':
    net = nn.Sequential(Flatten(), nn.Linear(784, 10))

elif args.model == 'MLP_face':
    dim_in = 512 * 4
    dim_hid = 32
    net = nn.Sequential(
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'MLP_face_128':
    dim_in = 512 * 4
    dim_hid = 128
    net = nn.Sequential(
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'MLP_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 32
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )

elif args.model == 'MLP_efficientnet_b0':
    dim_in = 1280 #* 7 * 7
    dim_hid = 32
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )

elif args.model == 'MLP256_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 256
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )

elif args.model == 'MLP512_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 512
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )

elif args.model == 'MLP128_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 128
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'MLP32_32_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 32
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'MLP64_64_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 64
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'CNN_efficientnet_b2':
    dim_in = 1408 #* 7 * 7
    dim_hid = 32
    
    net = nn.Sequential(
        nn.Conv2d(dim_in, 128, kernel_size = 1, stride = 1),
        nn.ReLU(),
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(128, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'MLP256_128_efficientnet_b0':
    dim_in = 1280 #* 7 * 7
    dim_hid = 128
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, 256),
        nn.ReLU(),
        nn.Linear(256, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )
elif args.model == 'MLP512_256_efficientnet_b0':
    dim_in = 1280 #* 7 * 7
    dim_hid = 256
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, 512),
        nn.ReLU(),
        nn.Linear(512, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )

elif args.model == 'MLP512_128_efficientnet_b0':
    dim_in = 1280 #* 7 * 7
    dim_hid = 128
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, 512),
        nn.ReLU(),
        nn.Linear(512, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )


elif args.model == 'MLP256_256_efficientnet_b0':
    dim_in = 1280 #* 7 * 7
    dim_hid = 256
    
    net = nn.Sequential(
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(dim_in, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, dim_hid),
        nn.ReLU(),
        nn.Linear(dim_hid, n_class),
    )











if len(args.pretrain_feat) > 0:
    pretrain_ckp = torch.load(args.pretrain_feat)
    pretrained = pretrain_ckp['net'].module
    pretrained = pretrained.to(device)
    pretrained.eval()

if len(args.efficientnet_pretrained) > 0:
    from efficientnet_pytorch import EfficientNet
    pretrained = EfficientNet.from_pretrained(args.efficientnet_pretrained)
    pretrained = pretrained.to(device)
    pretrained.eval()

    
net = net.to(device)

net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss(reduction='none')
opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.ExpWeightAvg > 0:
    Shadow_Params = [param.data.clone() for param in net.parameters()]



if args.toy_example_for_smooth_radius:
    param_dim = 0
    for param in net.parameters():
        param_dim += param.view(-1).size(0)
#    param_dim = 10000
    print ("param_dim: %d"%(param_dim))
    
    param = torch.cuda.FloatTensor(param_dim).normal_(0, 0.0001)
    for i in range(100):
        grad = -param
        L2_norm = float((grad * grad).sum() ** 0.5)
        coef = args.g_clip / max(args.g_clip, L2_norm)
        grad = grad * coef * args.batch_size * args.toy_example_for_smooth_radius_batch_coef
        param = param + args.lr * (grad + torch.cuda.FloatTensor(param_dim).normal_(0, args.noise_multiplier * float(args.g_clip)))
        
        if (i + 1) % 1 == 0:
            param_norm = float((param * param).sum() ** 0.5)
            
            std_noise = args.lr * torch.cuda.FloatTensor(param_dim).normal_(0, args.noise_multiplier * float(args.g_clip))
            std_noise_norm = float((std_noise * std_noise).sum() ** 0.5)


            smooth_radius = param_norm / std_noise_norm
            
            print ("[step: %d][param_norm: %.4f][anticipated_smooth_radius: %.4f]"%(i+1, param_norm, smooth_radius))
        
    exit()

for epoch in range(args.max_epoch):
    num_step = 0

    loss_train = 0
    acc_train = 0
    cnt_batch = 0
    cnt = 0

    _lr = float(args.lr * (args.lr_adj_ratio ** (epoch // args.lr_adj_period)))
    for param_group in opt.param_groups:
        param_group['lr'] = _lr

    for j, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        if len(args.pretrain_feat) > 0:
            with torch.no_grad():
                x = pretrained(x, feature_only = True)

        if len(args.efficientnet_pretrained) > 0:
            with torch.no_grad():
                x = pretrained.extract_features(x)

        if j % args.adaptive_period == 0 and args.adaptive_clipping:
            num_step += 1
        
            output = net(x)
            loss = criterion(output, y)
    
            dp_norm = 0.
            ndp_norm = []

            for i in range(0, min(batch_size, args.batch_size // 2)):
                opt.zero_grad()
                torch.autograd.backward(loss[i], retain_graph = True)
                _norm = 0.
                for param in net.parameters():
                    _norm += (param.grad.data**2).sum()
                _norm = float(_norm ** 0.5)
                ndp_norm.append(_norm)
                dp_norm += min(_norm, args.init_clip)

            dp_norm = dp_norm + torch.cuda.FloatTensor(1).normal_(0, args.noise_multiplier * float(args.init_clip))
            dp_norm = dp_norm / (args.batch_size // 2)

            if dp_norm > 0:
                args.g_clip = dp_norm
                args.init_clip = 10 * dp_norm
                if args.adaptive_lr:
                    _lr = float(args.lr / (dp_norm**args.adaptive_lr_T))
                    for param_group in opt.param_groups:
                        param_group['lr'] = _lr
            with open(logfile, 'a') as f:
                if args.debug:
                    print ("[epoch: %d][DP_norm: %.4f]"%(epoch, dp_norm))
                    print ("[actual norm][avg: %.4f][max: %.4f][min: %.4f]"%(np.mean(ndp_norm), np.max(ndp_norm), np.min(ndp_norm)))

                f.write("[epoch: %d][DP_norm: %.4f]\n"%(epoch, dp_norm))
                f.write("[actual norm][avg: %.4f][max: %.4f][min: %.4f]\n"%(np.mean(ndp_norm), np.max(ndp_norm), np.min(ndp_norm)))
            
            continue

        
        if args.debug:
            print("[epoch: %d][step: %d]"%(epoch, num_step+1))
        
        if args.mode == 'naive':
            cnt_batch += 1
            cnt += x.size(0)

            sum_grad = [torch.zeros_like(param) for param in net.parameters()]
            output = net(x)
            loss = criterion(output, y)
            
            for i in range(0, batch_size):
                opt.zero_grad()

                if args.L1_grad_coef > 0:
                    grad_i = torch.autograd.grad(loss[i], net.parameters(), retain_graph = True, create_graph = True, only_inputs = True)

                    L1_reg_i = 0.
                    for grad in grad_i:
                        L1_reg_i = L1_reg_i + grad.abs().mean()

                    torch.autograd.backward(L1_reg_i * args.L1_grad_coef, retain_graph=True)
                    
                    with torch.no_grad():
                        for param, grad in zip(net.parameters(), grad_i):
                            param.grad.data = param.grad.data + grad

                else:
                    torch.autograd.backward(loss[i], retain_graph = True)

                if args.grad_zero_out > 0:
                    for param in net.parameters():
                        _grads = param.grad.data.abs()
                        _line = torch.topk(_grads.view(-1), k = int(_grads.view(-1).size(0) * args.grad_zero_out), largest=False)[0].max()
                        param.grad.data[_grads < _line] = 0.
                        

                if (not args.no_privacy) or args.clip_no_privacy:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.g_clip, norm_type=2)
                sum_grad = [g + param.grad for param, g in zip(net.parameters(), sum_grad)]
                
            for param, g in zip(net.parameters(), sum_grad):
                if args.no_privacy:
                    param.grad.data = g
                else:
                    param.grad.data = (g + torch.cuda.FloatTensor(g.size()).normal_(0, args.noise_multiplier * float(args.g_clip)))

            opt.step()
            num_step += 1
            loss_train += torch.mean(loss).item()
            _, predicted = torch.max(output.data, 1)
            acc_train += predicted.eq(y.data).cpu().sum()
            if args.debug:
                print("[epoch %d][step: %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, num_step, loss_train / cnt_batch, float(acc_train) / cnt))


        elif args.mode == 'multi-batch':
            if args.momentum != 0:
                raise Exception("Potential privacy violation!")
            with torch.no_grad():
                original_param = [param.data.clone() for param in net.parameters()]
                sum_grad = [torch.zeros_like(param) for param in net.parameters()]

            
            for i in range(0, batch_size - args.T_multi + 1, args.T_multi):
                for param, o_param in zip(net.parameters(), original_param):
                    param.data = o_param.clone()
                
                for j in range(i, i+args.T_multi):
                    output = net(x[j:j+1])
                    loss = criterion(output, y[j:j+1])
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_train += loss.item()
                    _, predicted  = torch.max(output.data, 1)
                    acc_train += predicted.eq(y[j:j+1]).cpu().sum()
                    cnt += 1
                    cnt_batch += 1
                
                for param, o_param in zip(net.parameters(), original_param):
                    param.grad.data = param.data - o_param.data
                    
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.g_clip, norm_type=2)
                sum_grad = [g + param.grad.data for param, g in zip(net.parameters(), sum_grad)]

            for (param, o_param), g in zip(zip(net.parameters(), original_param), sum_grad):
                param.data = o_param + g + torch.cuda.FloatTensor(g.size()).normal_(0, args.noise_multiplier * float(args.g_clip))

            num_step += 1
            if args.debug:
                print("[epoch %d][step: %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, num_step, loss_train / cnt_batch, float(acc_train) / cnt))

#--------------------------------------------------------------------------------------
        elif args.mode == 'smooth_loss':
            cnt_batch += 1
            cnt += x.size(0)

            with torch.no_grad():
                original_param = [param.data.clone() for param in net.parameters()]
                sum_grad = [[torch.zeros_like(param) for param in net.parameters()] for _ in range(0, batch_size) ]

            for sl_idx in range(0, args.smooth_loss_samples):
                #compute perturbed parameters
                for param, o_param in zip(net.parameters(), original_param):
                    param.data = o_param + args.smooth_loss_radius * args.lr * torch.cuda.FloatTensor(param.size()).normal_(0, args.noise_multiplier * float(args.g_clip))

                output = net(x)
                loss = criterion(output, y)
            
                for i in range(0, batch_size):
                    opt.zero_grad()

                    torch.autograd.backward(loss[i], retain_graph = True)
                    sum_grad[i] = [g + param.grad for param, g in zip(net.parameters(), sum_grad[i])]
            
            for param, o_param in zip(net.parameters(), original_param):
                param.data = o_param

            final_grad = [torch.zeros_like(param) for param in net.parameters()]
            for i in range(0, batch_size):
                #clip sum_grad[i]
                #sum sum_grad[] into sum_grad

                L2_norm = 0.
                for g in sum_grad[i]:
                    L2_norm += (g * g).sum()
                L2_norm = float(L2_norm ** 0.5)
                if args.debug and i == 0:
                    print (L2_norm)
                coef = args.g_clip / max(args.g_clip, L2_norm)
                final_grad = [fg + g * coef for fg, g in zip(final_grad, sum_grad[i])]
            

            for param, g in zip(net.parameters(), final_grad):
                param.grad.data = (g + torch.cuda.FloatTensor(g.size()).normal_(0, args.noise_multiplier * float(args.g_clip)))

            opt.step()
            num_step += 1
            loss_train += torch.mean(loss).item()
            _, predicted = torch.max(output.data, 1)
            acc_train += predicted.eq(y.data).cpu().sum()
            if args.debug:
                print("[epoch %d][step: %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, num_step, loss_train / cnt_batch, float(acc_train) / cnt))
            
#--------------------------------------------------------------------------------------

        

        if args.ExpWeightAvg > 0:
            with torch.no_grad():
                for (param, s_param) in zip(net.parameters(), Shadow_Params):
                    s_param.data = s_param.data * args.ExpWeightAvg + param.data * (1 - args.ExpWeightAvg)
    
        if args.frequent_ckp_last10epoch and epoch + 10 >= args.max_epoch:
            torch.save({'net':net}, './checkpoint/' + args.runname+'_frequent_' + str(frequent_ckp_cnt) + '.t7')
            frequent_ckp_cnt += 1
        elif args.frequent_ckp_last20epoch and epoch + 20 >= args.max_epoch:
            torch.save({'net':net}, './checkpoint/' + args.runname+'_frequent_' + str(frequent_ckp_cnt) + '.t7')
            frequent_ckp_cnt += 1
    
    acct.compose_poisson_subsampled_mechanisms(func, batch_sampler.rate, num_step)
    eps_now = acct.get_eps(args.delta)
    
    with open(logfile, 'a') as f:
        if args.debug:
            print("[epoch %d][eps: %.3f][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, eps_now, loss_train / cnt_batch, float(acc_train) / cnt))
        f.write("[epoch %d][eps: %.3f][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, eps_now, loss_train / cnt_batch, float(acc_train) / cnt))


    loss_test = 0
    acc_test = 0
    cnt_batch = 0
    cnt = 0

    if args.ExpWeightAvg > 0:
        with torch.no_grad():
            for (param, s_param) in zip(net.parameters(), Shadow_Params):
                param.data = s_param.data

    for x, y in test_loader:
        cnt_batch += 1
        x, y = x.to(device), y.to(device)
        
        if len(args.pretrain_feat) > 0:
            with torch.no_grad():
                x = pretrained(x, feature_only = True)

        if len(args.efficientnet_pretrained) > 0:
            with torch.no_grad():
                x = pretrained.extract_features(x)

        with torch.no_grad():
            output = net(x)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, y)
            loss_test += torch.mean(loss).item()
            acc_test += predicted.eq(y.data).cpu().sum()
            cnt += x.size(0)
    with open(logfile, 'a') as f:
        if args.debug:
            print("[epoch %d][eps: %.3f][Loss_test: %.3f][Acc_test: %.4f]\n" % (epoch, eps_now, loss_test / cnt_batch, float(acc_test) / cnt))
        f.write("[epoch %d][eps: %.3f][Loss_test: %.3f][Acc_test: %.4f]\n" % (epoch, eps_now, loss_test / cnt_batch, float(acc_test) / cnt))

    if (epoch + 1) % args.save_epoch == 0:
        
        torch.save({'net':net, 'eps':eps_now}, './checkpoint/' + args.runname + '_' + str(epoch) + '.t7')

    














