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
best_prec1 = 0
encoder_checkpoint_path = './Models/Encoder_src.pt'
config_file='./tvarit_wheel.yaml'
model = None
optimizer_a = None
optimizer_c = None
vnet = None

class YAML:
    def __init__(self, data):
        self.data = data
    
    def __getattr__(self, item):
        return self.data[item]

def parse_configs_and_args(config_path='tvarit_wheel.yaml'):
    global config_file
    # parse arguments
    parser = argparse.ArgumentParser(description='Imbalanced Example')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_meta', type=int, default=50,
                        help='The number of meta data for each class.')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--split', type=int, default=1000)
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

def train(train_loader, validation_loader, encoder, epoch, configs):
    """Train for one epoch on the training set"""
    global model, vnet, optimizer_a, optimizer_c
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    #wfl_loss = WeightedFocalLoss()
    model.train()

    for i, (input, target,_,_) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)
        target_var = target_var.view(-1)

        #Create a replica of the classifier model
        meta_model = build_model(configs)
        meta_model.load_state_dict(model.state_dict())

        #Use the meta-model to identify cost for each train sample and then use cost obtain the weight.
        #With the weights, calculate loss for this meta-model and update its parameters
        y_f_hat = meta_model(encoder(input_var))
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)#wfl_loss(y_f_hat, target_var)
        cost_v = torch.reshape(cost, (len(cost), 1))

        v_lambda = vnet(cost_v)

        norm_c = torch.sum(v_lambda)

        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda

        l_f_meta = torch.sum(cost_v * v_lambda_norm)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads
        #At the end, Meta-model has taken a training step with complete data (batch).

        #Use the updated meta-model to compute loss of VNet on balanced dataset and update the VNet parameters
        (input_validation, target_validation,_,_) = next(iter(validation_loader))
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)
        target_validation_var = target_validation_var.view(-1)
        y_g_hat = meta_model(encoder(input_validation_var))
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)#wfl_loss(y_g_hat, target_validation_var).mean()
        prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        optimizer_c.step()
        #At the end, weight network has taken a step with the smaller balanced data.

        #Now finally, use the updated VNet and update the parameters of the actual classifier model
        y_f = model(encoder(input_var))
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)#wfl_loss(y_f, target_var)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)
        norm_v = torch.sum(w_new)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        l_f = torch.sum(cost_v * w_v)

        losses.update(l_f.item(), input.size(0))
        meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader),
                loss=losses,meta_loss=meta_losses, top1=top1,meta_top1=meta_top1))
    configs['wandb'].log({'Train Loss':losses.avg})
    configs['wandb'].log({'Meta Train Loss':meta_losses.avg})
    train_auc_score = get_auc_score(train_loader, encoder)
    meta_train_auc_score = get_auc_score(validation_loader, encoder)
    configs['wandb'].log({'Train AUC':train_auc_score})
    configs['wandb'].log({'Meta Train AUC':meta_train_auc_score})
    print('Train_AUC {train_auc:.3f}\t'
          'Meta_Train_AUC {meta_train_auc:.3f}'.format(train_auc=train_auc_score,meta_train_auc=meta_train_auc_score))


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
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    val_auc_score = get_auc_score(val_loader, encoder)
    print(' * Val_AUC {val_auc:.3f}'.format(val_auc=val_auc_score))
    configs['wandb'].log({'Validation Loss':losses.avg})
    configs['wandb'].log({'Val AUC':val_auc_score})
    return top1.avg

#Create the encoder and classifier models. Encoder is a LSTM network and Classifier is an MLP network. 
def build_model(global_configs, meta_model=True):
    global encoder_checkpoint_path, args
    Encoder_src = Encoder_LSTM(n_features=global_configs['Source_no_of_features'], embedding_dim=global_configs['latent_dim'],N=global_configs['enc_layers'])
    Encoder_src.load_state_dict(torch.load(encoder_checkpoint_path))
    f1,f2,f3 = 256,128,64
    Classifier = MLP_model(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim'])

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

#Main function that runs the code. Parses the arguments passed in the command line and also from the yaml file containing arguments
def main():
    global args, best_prec1, model, optimizer_a, optimizer_c, vnet

    #Parse the args and the configs. Args -> command line inputs, configs -> yaml file inputs
    args, configs = parse_configs_and_args()
    #Prepare the unbiased/balanced meta-data for the MW network from the train loader.
    configs['Meta_Data_Loader'], configs['Source_Train_Loader'] = prepare_meta_data(configs, args.num_meta)
    
    # create models
    Encoder, model = build_model(configs, meta_model=False)

    #Initiating wandb project to log the training regime dynamics
    wandb.init(project="Class Imabalance")
    wandb.watch(model, log_freq=100)
    wandb.run.name = configs['jobid']  
    configs['wandb'] = wandb  

    #Freeze the encoder model
    for param in Encoder.parameters():
        param.requires_grad = False
    Encoder = Encoder.eval()

    #Prepare the optimizer for the classifier network. Encoder is frozen, hence no backprop needed through it
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    #Initialize the Weights Network and its optimizer
    vnet = VNet(1, 100, 1).cuda()

    optimizer_c = torch.optim.SGD(vnet.params(), 1e-5,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = WeightedFocalLoss().cuda()

    for epoch in range(configs['epochs']):
        adjust_learning_rate(optimizer_a, epoch + 1)

        #Command to train the model for 1 epoch
        train(configs['Source_Train_Loader'], configs['Meta_Data_Loader'],Encoder, epoch, configs)

        # evaluate on validation set
        prec1 = validate(configs['Source_Eval_Loader'], Encoder, criterion, epoch, configs)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch % 10 == 0: #Prepare checkpoints at every 10th epoch and save them
            ckpt_folder = './checkpoints/' + configs['jobid']
            if not os.path.exists(ckpt_folder):
                os.mkdir(ckpt_folder)
            ckpt_path = os.path.join(ckpt_folder, 'model_epoch_%d.pt' % (epoch))
            torch.save(model.state_dict(), ckpt_path)

    print('Best accuracy: ', best_prec1)
    torch.save(model.state_dict(), './checkpoints/' + configs['jobid'] + '/final_model.pt')

if __name__ == '__main__':
    main()




