import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from data_utils import *
from resnet import *
from raw_tvarit import *
from utilis import *
import yaml
import wandb
import warnings
warnings.filterwarnings("ignore")


args = None
encoder_checkpoint_path = './Models/Encoder_src.pt'
config_file='./tvarit_wheel.yaml'
model = None
optimizer = None

class YAML:
    def __init__(self, data):
        self.data = data
    
    def __getattr__(self, item):
        return self.data[item]

def parse_configs_and_args(config_path='tvarit_wheel.yaml'):
    global config_file
    # parse arguments
    parser = argparse.ArgumentParser(description='Imbalanced Example')
    # parser.add_argument('--dataset', default='cifar10', type=str,
    #                     help='dataset (cifar10 [default] or cifar100)')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_meta', type=int, default=50,
                        help='The number of meta data for each class.')
    # parser.add_argument('--imb_factor', type=float, default=0.1)
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    # parser.add_argument('--epochs', type=int, default=500, metavar='N',
    #                     help='number of epochs to train')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--split', type=int, default=1000)
    # parser.add_argument('--seed', type=int, default=42, metavar='S',
    #                     help='random seed (default: 42)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    args = parser.parse_args()

    with open(config_file, mode='r') as f:
        tvarit_args = YAML(yaml.load(f, Loader=yaml.FullLoader))
    global_configs = get_tvarit_raw(tvarit_args)

    return args, global_configs

def get_auc_score(data_loader, encoder):
    global model
    model.eval()

    target_list = []
    pred_list = []
    for i, (data, targets, _, _) in enumerate(data_loader):
        data,targets = data.cuda(), targets.cuda()
        data = encoder(data)
        targets = targets.view(-1)
        logits = model(data)
        preds = torch.softmax(logits, dim=1)
        preds = preds[:,1]
        target_temp = targets.cpu().detach().numpy().tolist()
        preds_temp = preds.cpu().detach().numpy().tolist()

        target_list.extend(target_temp)
        pred_list.extend(preds_temp)

    auc_val = sm.roc_auc_score(np.array(target_list), np.array(pred_list))
    return auc_val

def train(train_loader, encoder, epoch, configs):
    """Train for one epoch on the training set"""
    global model, optimizer
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    for i, (input, target,_,_) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)
        target_var = target_var.view(-1)
        
        y_f = model(encoder(input_var))
        l_f = F.cross_entropy(y_f, target_var)
        
        losses.update(l_f.item(), input.size(0))
        
        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                loss=losses))
    configs['wandb'].log({'Train Loss':losses.avg})
    train_auc_score = get_auc_score(train_loader, encoder)
    configs['wandb'].log({'Train AUC':train_auc_score})
    print('Train_AUC {train_auc:.3f}\t'.format(train_auc=train_auc_score))


def validate(val_loader, encoder, criterion, epoch, configs):
    """Perform validation on the validation set"""
    global model
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target,_,_) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.view(-1)

        # compute output
        with torch.no_grad():
            output = model(encoder(input_var))
        loss = criterion(output, target_var)
        #loss = criterion(output, target_var).mean()

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
           # configs['wandb'].log({'Validation Loss':losses.avg})
            #val_auc_score = get_auc_score(val_loader, encoder)
            #configs['wandb'].log({'Validation AUC':val_auc})
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

    val_auc_score = get_auc_score(val_loader, encoder)
    print(' * Val_AUC {val_auc:.3f}'.format(val_auc=val_auc_score))
    configs['wandb'].log({'Validation Loss':losses.avg})
    configs['wandb'].log({'Val AUC':val_auc_score})
    # log to TensorBoard

    return


def build_model(global_configs, meta_model=True):
    global encoder_checkpoint_path, args
    Encoder_src = Encoder_LSTM(n_features=global_configs['Source_no_of_features'], embedding_dim=global_configs['latent_dim'],N=global_configs['enc_layers'])
    Encoder_src.load_state_dict(torch.load(encoder_checkpoint_path))
    f1,f2,f3 = 256,128,64#args.def_f1,args.def_f2,args.def_f3
    Classifier = MLP_model(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim'])
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        Encoder_src.cuda()
        Classifier.cuda()
        torch.backends.cudnn.benchmark = True

    if meta_model:
        return Classifier
    else:
        return Encoder_src, Classifier

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    global args, model, optimizer

    args, configs = parse_configs_and_args()
    #Prepare the unbiased meta-data for the MW network
    #print(args.num_meta)
    #exit()
    # configs['Meta_Data_Loader'], configs['Source_Train_Loader'] = prepare_meta_data(configs, args.num_meta)
    #Adjust the imbalance in the val dataset - PENDING
    
    # create model
    Encoder, model = build_model(configs, meta_model=False)


    wandb.init(project="Class Imabalance")
    wandb.watch(model, log_freq=100)
    wandb.run.name = configs['jobid']  
    configs['wandb'] = wandb  
    #Freeze the encoder model
    for param in Encoder.parameters():
        param.requires_grad = False

    Encoder = Encoder.eval()
    optimizer = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = WeightedFocalLoss().cuda()

    for epoch in range(configs['epochs']):
        adjust_learning_rate(optimizer, epoch + 1)

        train(configs['Source_Train_Loader'],Encoder, epoch, configs)

        # evaluate on validation set
        validate(configs['Source_Eval_Loader'], Encoder, criterion, epoch, configs)

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)

        if epoch % 10 == 0:
            ckpt_folder = './checkpoints/' + configs['jobid']
            if not os.path.exists(ckpt_folder):
                os.mkdir(ckpt_folder)
            ckpt_path = os.path.join(ckpt_folder, 'model_epoch_%d.pt' % (epoch))
            torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), './checkpoints/' + configs['jobid'] + '/final_model.pt')
    train_auc = get_auc_score(configs['Source_Train_Loader'], Encoder)
    val_auc = get_auc_score(configs['Source_Eval_Loader'], Encoder)
    test_auc = get_auc_score(configs['Source_Test_Loader'], Encoder)

    print('Train_AUC {train_auc:.3f}\t'
          'Val_AUC {val_auc:.3f}\t'
          'Test_AUC {test_auc:.3f}\t'.format(train_auc=train_auc,val_auc=val_auc,test_auc=test_auc))
    
    f1 = open("auc_values.txt","a")
    f1.write(f"Train AUC: {train_auc}\nVal AUC: {val_auc}\nTest AUC: {test_auc}\n")
    f1.close()


if __name__ == '__main__':
    main()




