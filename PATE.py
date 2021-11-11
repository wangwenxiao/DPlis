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
from utils.data import DataSplit
import utils.PATE_core as PATE_core
import utils.PATE_smooth_sensitivity as PATE_ss
import math
import sys
from model.model_dptrain import SmallCNN_CIFAR as CNN


import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--data', type=str, default='SVHN')
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--runname', type=str, default="noname")
parser.add_argument('--delta', type=float, default=1e-6)
parser.add_argument('--mode', type=str, default='naive', help='mode in [naive, smooth_loss')

parser.add_argument('--smooth_loss_samples', type=int, default=1)
parser.add_argument('--smooth_loss_radius', type=float, default=1.0)
parser.add_argument('--smooth_prop_to_lr', action='store_true')

parser.add_argument('--lr_adj_period', type=int, default=99999)
parser.add_argument('--lr_adj_ratio', type=float, default=1.)

parser.add_argument('--g_clip', type=float, default=1.)
parser.add_argument('--clip', action='store_true')


#param for Confident-GNMax in PATE
parser.add_argument('--T', type=float, default=300)
parser.add_argument('--sigma_1', type=float, default=200)
parser.add_argument('--sigma_2', type=float, default=40)
parser.add_argument('--data_ind', action='store_true')

parser.add_argument('--path_teacher_label', type=str, default='./PATE_labels/svhn_250_teachers_labels.npy')
parser.add_argument('--train_N', type=int, default=0)
parser.add_argument('--test_N', type=int, default=10000)




parser.add_argument('--save_epoch', type=int, default=99999)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug2', action='store_true')

parser.add_argument('--smooth_sensitivity', action='store_true')

parser.add_argument('--model', type=str, default='CNN')




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

if args.data == 'SVHN':
    root = os.path.expanduser('~/data')
    
    original_test_set = datasets.SVHN(root, split='test', download=True)
    original_N = len(original_test_set)
    n_class = 10

#compute mean&std on train set
if args.train_N > 0:
    train_N = args.train_N
    test_N = args.test_N
else:
    train_N = original_N // 2
    test_N = original_N - train_N


mean = torch.zeros(3)
for i in range(train_N):
    img = transforms.ToTensor()(original_test_set[i][0])
    mean += img.mean((1, 2))

mean /= train_N
std_dev = torch.zeros(3)
for i in range(train_N):
    img = transforms.ToTensor()(original_test_set[i][0])
    std_dev += ((img - mean.view(3, 1, 1))**2).mean((1, 2))

std_dev /= train_N
std_dev = std_dev.sqrt()

print ("original_N:", original_N)
print ("train_N:", train_N)
print ("test_N:", test_N)
print ("mean:", mean)
print ("std_dev:", std_dev)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std_dev),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std_dev),
])

train_set = DataSplit(original_test_set, 0, train_N)
test_set = DataSplit(original_test_set, original_N - test_N, original_N, transform_test)

def _compute_rdp(votes, threshold, sigma1, sigma2, delta, orders, data_ind):
    """Computes the (data-dependent) RDP curve for Confident GNMax."""
    rdp_cum = np.zeros(len(orders))
    rdp_sqrd_cum = np.zeros(len(orders))
    answered = 0

    for i, v in enumerate(votes):
        if threshold is None:
            logq_step1 = 0  # No thresholding, always proceed to step 2.
            rdp_step1 = np.zeros(len(orders))
        else:
            logq_step1 = PATE_core.compute_logpr_answered(threshold, sigma1, v)
            if data_ind:
                rdp_step1 = PATE_core.compute_rdp_data_independent_threshold(sigma1, orders)
            else:
                rdp_step1 = PATE_core.compute_rdp_threshold(logq_step1, sigma1, orders)

        if data_ind:
            rdp_step2 = PATE_core.rdp_data_independent_gaussian(sigma2, orders)
        else:
            logq_step2 = PATE_core.compute_logq_gaussian(v, sigma2)
            rdp_step2 = PATE_core.rdp_gaussian(logq_step2, sigma2, orders)

        q_step1 = np.exp(logq_step1)
        rdp = rdp_step1 + rdp_step2 * q_step1
        # The expression below evaluates
        #         E[(cost_of_step_1 + Bernoulli(pr_of_step_2) * cost_of_step_2)^2]
        rdp_sqrd = (
                rdp_step1**2 + 2 * rdp_step1 * q_step1 * rdp_step2 +
                q_step1 * rdp_step2**2)
        rdp_sqrd_cum += rdp_sqrd

        rdp_cum += rdp
        answered += q_step1
        if ((i + 1) % 1000 == 0) or (i == votes.shape[0] - 1):
            rdp_var = rdp_sqrd_cum / i - (
                    rdp_cum / i)**2    # Ignore Bessel's correction.
            eps_total, order_opt = PATE_core.compute_eps_from_delta(orders, rdp_cum, delta)
            order_opt_idx = np.searchsorted(orders, order_opt)
            eps_std = ((i + 1) * rdp_var[order_opt_idx])**.5    # Std of the sum.
            print(
                    'queries = {}, E[answered] = {:.2f}, E[eps] = {:.3f} (std = {:.5f}) '
                    'at order = {:.2f} (contribution from delta = {:.3f})'.format(
                            i + 1, answered, eps_total, eps_std, order_opt,
                            -math.log(delta) / (order_opt - 1)))
            sys.stdout.flush()

        _, order_opt = PATE_core.compute_eps_from_delta(orders, rdp_cum, delta)

    return order_opt, eps_total

def _is_data_ind_step1(num_teachers, threshold, sigma1, orders):
    if threshold is None:
        return True
    return np.all(PATE_core.is_data_independent_always_opt_threshold(num_teachers, threshold, sigma1, orders))

def _is_data_ind_step2(num_teachers, num_classes, sigma, orders):
    return np.all(PATE_core.is_data_independent_always_opt_gaussian(num_teachers, num_classes, sigma, orders))

def _check_conditions(sigma, num_classes, orders):
    """Symbolic-numeric verification of conditions C5 and C6.
    The conditions on the beta function are verified by constructing the beta
    function symbolically, and then checking that its derivative (computed
    symbolically) is non-negative within the interval of conjectured monotonicity.
    The last check is performed numerically.
    """

    print('Checking conditions C5 and C6 for all orders.')
    sys.stdout.flush()
    conditions_hold = True

    for order in orders:
        cond5, cond6 = PATE_ss.check_conditions(sigma, num_classes, order)
        conditions_hold &= cond5 and cond6
        if not cond5:
            print('Condition C5 does not hold for order =', order)
        elif not cond6:
            print('Condition C6 does not hold for order =', order)

    if conditions_hold:
        print('Conditions C5-C6 hold for all orders.')
    sys.stdout.flush()
    return conditions_hold


def _find_optimal_smooth_sensitivity_parameters(votes, num_teachers, threshold, sigma1, sigma2, delta, ind_step1, ind_step2, order):
    """Optimizes smooth sensitivity parameters by minimizing a cost function.
    The cost function is
                exact_eps + cost of GNSS + two stds of noise,
    which captures that upper bound of the confidence interval of the sanitized
    privacy budget.
    Since optimization is done with full view of sensitive data, the results
    cannot be released.
    """
    rdp_cum = 0
    answered_cum = 0
    ls_cum = 0

    # Define a plausible range for the beta values.
    betas = np.arange(.3 / order, .495 / order, .01 / order)
    cost_delta = math.log(1 / delta) / (order - 1)

    for i, v in enumerate(votes):
        if threshold is None:
            log_pr_answered = 0
            rdp1 = 0
            ls_step1 = np.zeros(num_teachers)
        else:
            log_pr_answered = PATE_core.compute_logpr_answered(threshold, sigma1, v)
            if ind_step1:    # apply data-independent bound for step 1 (thresholding).
                rdp1 = PATE_core.compute_rdp_data_independent_threshold(sigma1, order)
                ls_step1 = np.zeros(num_teachers)
            else:
                rdp1 = PATE_core.compute_rdp_threshold(log_pr_answered, sigma1, order)
                ls_step1 = PATE_ss.compute_local_sensitivity_bounds_threshold(v, num_teachers, threshold, sigma1, order)

        pr_answered = math.exp(log_pr_answered)
        answered_cum += pr_answered

        if ind_step2:    # apply data-independent bound for step 2 (GNMax).
            rdp2 = PATE_core.rdp_data_independent_gaussian(sigma2, order)
            ls_step2 = np.zeros(num_teachers)
        else:
            logq_step2 = PATE_core.compute_logq_gaussian(v, sigma2)
            rdp2 = PATE_core.rdp_gaussian(logq_step2, sigma2, order)
            # Compute smooth sensitivity.
            ls_step2 = PATE_ss.compute_local_sensitivity_bounds_gnmax(
                    v, num_teachers, sigma2, order)

        rdp_cum += rdp1 + pr_answered * rdp2
        ls_cum += ls_step1 + pr_answered * ls_step2    # Expected local sensitivity.

        if ind_step1 and ind_step2:
            # Data-independent bounds.
            cost_opt, beta_opt, ss_opt, sigma_ss_opt = None, 0., 0., np.inf
        else:
            # Data-dependent bounds.
            cost_opt, beta_opt, ss_opt, sigma_ss_opt = np.inf, None, None, None

            for beta in betas:
                ss = PATE_ss.compute_discounted_max(beta, ls_cum)

                # Solution to the minimization problem:
                #     min_sigma {order * exp(2 * beta)/ sigma^2 + 2 * ss * sigma}
                sigma_ss = ((order * math.exp(2 * beta)) / ss)**(1 / 3)
                cost_ss = PATE_ss.compute_rdp_of_smooth_sensitivity_gaussian(beta, sigma_ss, order)

                # Cost captures exact_eps + cost of releasing SS + two stds of noise.
                cost = rdp_cum + cost_ss + 2 * ss * sigma_ss
                if cost < cost_opt:
                    cost_opt, beta_opt, ss_opt, sigma_ss_opt = cost, beta, ss, sigma_ss

        if ((i + 1) % 100 == 0) or (i == votes.shape[0] - 1):
            eps_before_ss = rdp_cum + cost_delta
            eps_with_ss = (
                    eps_before_ss + PATE_ss.compute_rdp_of_smooth_sensitivity_gaussian(
                            beta_opt, sigma_ss_opt, order))
            print('{}: E[answered queries] = {:.1f}, RDP at {} goes from {:.3f} to '
                        '{:.3f} +/- {:.3f} (ss = {:.4}, beta = {:.4f}, sigma_ss = {:.3f})'.
                        format(i + 1, answered_cum, order, eps_before_ss, eps_with_ss,
                                     ss_opt * sigma_ss_opt, ss_opt, beta_opt, sigma_ss_opt))
            sys.stdout.flush()

    # Return optimal parameters for the last iteration.
    return beta_opt, ss_opt, sigma_ss_opt

class ConfidentGNMax(torch.utils.data.Dataset):
    def __init__(self, dataset, T, sigma_1, sigma_2, path_teacher_label, n_class, delta, transform=None):
        self.data = []
        self.path_teacher_label = path_teacher_label
        self.teacher_label = np.load(path_teacher_label)
        self.n_teacher = self.teacher_label.shape[0]
        self.transform = transform

        cnt_correct = 0

        np.random.seed(19980208)

        orders = np.concatenate((np.arange(2, 100 + 1, .05), 
            np.logspace(np.log10(100), np.log10(500),num=100)))
   
        cnt = np.zeros((len(dataset), n_class))
        
        for i in range(len(dataset)):
            _cnt = np.bincount(self.teacher_label[:, i])
            cnt[i, :_cnt.shape[0]] = _cnt
            
            
            if cnt[i].max() + np.random.normal(0, sigma_1) >= T:
                noisy_label = np.argmax(cnt[i] + np.random.normal(0, sigma_2, cnt[i].shape))
                
                self.data.append((dataset[i][0], noisy_label))
                cnt_correct += noisy_label == dataset[i][1]

        if args.debug2:
            return

        data_ind = args.data_ind
        order, eps = _compute_rdp(cnt, T, sigma_1, sigma_2, delta, orders, data_ind = data_ind)

        if args.smooth_sensitivity:
            ind_step1 = _is_data_ind_step1(self.n_teacher, T, sigma_1, order)
            ind_step2 = _is_data_ind_step2(self.n_teacher, n_class, sigma_2, order)
            if not (data_ind or (ind_step1 and ind_step2)):
                if _check_conditions(sigma_2, n_class, [order]):
                    beta_opt, ss_opt, sigma_ss_opt = _find_optimal_smooth_sensitivity_parameters(cnt, self.n_teacher, T, sigma_1, sigma_2, delta, ind_step1, ind_step2, order)
                    print('Optimal beta = {:.4f}, E[SS_beta] = {:.4}, sigma_ss = {:.2f}'.format(beta_opt, ss_opt, sigma_ss_opt)) 
                
        
            
        print ("PATE_train_N:", len(self.data))
        print ("label acc:", float(cnt_correct) / len(self.data))
        print ("eps, delta:", eps, delta)
        with open(logfile, 'a') as f:
            f.write("[PATE_train_N: %d][label acc: %.3f]\n"%(len(self.data), float(cnt_correct) / len(self.data)))
            f.write("[eps: %.3f][delta: %.10f]\n"%(eps, delta))
                
        
        

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

PATE_train_set = ConfidentGNMax(train_set, args.T, args.sigma_1, args.sigma_2, args.path_teacher_label, n_class, args.delta, transform_train)

device = 'cuda'
if args.model == 'CNN':
    net = CNN(n_class, 3)

net = net.to(device)

net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

train_loader = torch.utils.data.DataLoader(PATE_train_set, batch_size=args.batch_size, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


for epoch in range(args.max_epoch):
    num_step = 0

    loss_train = 0
    acc_train = 0
    cnt_batch = 0
    cnt = 0
    net.train()
    
    _lr = float(args.lr * (args.lr_adj_ratio ** (epoch // args.lr_adj_period)))
    for param_group in opt.param_groups:
        param_group['lr'] = _lr

    if args.smooth_prop_to_lr:
        args.smooth_loss_radius *= _lr

    for j, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        
        cnt_batch += 1
        cnt += x.size(0)

        if args.mode == 'naive':

            output = net(x)
            loss = criterion(output, y)
        
            opt.zero_grad()
            loss.backward()
            
            if args.clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.g_clip, norm_type=2.0)

            opt.step()
        else:
            with torch.no_grad():
                original_param = [param.data.clone() for param in net.parameters()]
                sum_grad = [torch.zeros_like(param) for param in net.parameters()]
            
            for sl_idx in range(0, args.smooth_loss_samples):
                for param, o_param in zip(net.parameters(), original_param):
                    param.data = o_param + torch.cuda.FloatTensor(param.size()).normal_(0, args.smooth_loss_radius)
                output = net(x)
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                
                sum_grad = [g + param.grad for param, g in zip(net.parameters(), sum_grad)]

            for param, o_param in zip(net.parameters(), original_param):
                param.data = o_param
                
            for param, g in zip(net.parameters(), sum_grad):
                param.grad.data = g / args.smooth_loss_samples

            if args.clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.g_clip, norm_type=2.0)
                
            opt.step()




        num_step += 1
        loss_train += torch.mean(loss).item()
        _, predicted = torch.max(output.data, 1)
        acc_train += predicted.eq(y.data).cpu().sum()
        #if args.debug:
        #    print("[epoch %d][step: %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, num_step, loss_train / cnt_batch, float(acc_train) / cnt))

    with open(logfile, 'a') as f:
        if args.debug:
            print("[epoch %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, loss_train / cnt_batch, float(acc_train) / cnt))
        f.write("[epoch %d][Loss_train: %.3f][Acc_train: %.4f]\n" % (epoch, loss_train / cnt_batch, float(acc_train) / cnt))

    if args.smooth_prop_to_lr:
        args.smooth_loss_radius /= _lr


    loss_test = 0
    acc_test = 0
    cnt_batch = 0
    cnt = 0
    net.eval()

    for x, y in test_loader:
        cnt_batch += 1
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            output = net(x)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, y)
            loss_test += torch.mean(loss).item()
            acc_test += predicted.eq(y.data).cpu().sum()
            cnt += x.size(0)

    with open(logfile, 'a') as f:
        if args.debug:
            print("[epoch %d][Loss_test: %.3f][Acc_test: %.4f]\n" % (epoch, loss_test / cnt_batch, float(acc_test) / cnt))
        f.write("[epoch %d][Loss_test: %.3f][Acc_test: %.4f]\n" % (epoch, loss_test / cnt_batch, float(acc_test) / cnt))

    if (not args.debug) and ((epoch + 1) % args.save_epoch == 0 or epoch + 1 == args.max_epoch):
        torch.save({'net':net}, './checkpoint/' + args.runname + '_' + str(epoch) + '.t7')





