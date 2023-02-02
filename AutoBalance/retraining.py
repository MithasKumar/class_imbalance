import argparse
import os
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

import torch

from utils.metrics import print_num_params

from core.trainer import train_epoch, eval_epoch, evaluate_model_tvarit
from core.utils import loss_adjust_cross_entropy, cross_entropy
from core.utils import get_init_dy, get_init_ly, get_train_w, get_val_w

from dataset.ImageNet_LT import ImageNetLTDataLoader
from dataset.cifar10 import load_cifar10
from dataset.iNaturalist import INAT
from dataset.cifar100 import load_cifar100

import torchvision.models as models
from models.ResNet import ResNet32

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


## Tvarit data imports

import yaml
from wheel_exp_main import get_models
from dataset.raw_tvarit import get_tvarit_raw

import wandb

class YAML:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, item):
        return self.data[item]

assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config',
                    default="./results/tvarit_wheel/New_Experiments/test_big_data/dyly_LA_init/1/config.yaml", type=str)
parser.add_argument('--model_type', dest='model_type', default="encoder", type=str)
parser.add_argument('--lower_lr', dest='lower_lr', default=10**(-4), type=float)

args = parser.parse_args()

model_type = args.model_type
lower_level_lr = args.lower_lr

with open(args.config, mode='r') as f:
    args = yaml.load(f, Loader=yaml.UnsafeLoader)

args['model_type'] = model_type

assert os.path.exists(args["save_path"])
args["up_configs"]["dy"]=False
args["up_configs"]["ly"]=False
args["up_configs"]["wy"]=False
args["up_configs"]["aug"]=False
args["up_configs"]["ly_init"]="Retrain"
args["up_configs"]["dy_init"]="Retrain"
args["up_configs"]["wy_init"]="Retrain"
args["up_configs"]["aug_init"]="Retrain"
args["up_configs"]["start_epoch"]=10000
args["up_configs"]["end_epoch"]=-1
print(args["up_configs"]["ly_LA_tau"],args["group_size"])
if args["up_configs"]["ly_LA_tau"]==None:
    args["group_size"]=1
print(args["up_configs"]["ly_LA_tau"],args["group_size"])
device = args["device"]
dataset = args["dataset"]

if dataset == 'tvarit':

    wandb.init(project="Class Imabalance", mode='disabled')
    num_classes = 2
    config_file = './configs/Wheel/tvarit_wheel.yaml'

    # with open(r'/home/ece/Lakshya/Projects/prj_imb/AutoBalance/configs/Wheel/tvarit_wheel.yaml') as file:
    #     tvarit_args = yaml.load(file, Loader=yaml.FullLoader)

    with open(config_file, mode='r') as f:
        tvarit_args = YAML(yaml.load(f, Loader=yaml.FullLoader))

    print(tvarit_args)

    global_configs, Encoder_src, model = get_models(tvarit_args, model_type = args['model_type'])

    model = model.to(device)

    wandb.watch(model, log_freq=100)
    # wandb.run.name = exp_name

    args['wandb'] = wandb

    if args['model_type'] == 'encoder':
        Encoder_src = Encoder_src.to(device)

    train_loader = global_configs['Source_Train_Loader']
    val_loader = global_configs['Source_Eval_Loader']
    test_loader = global_configs['Source_Test_Loader']

    eval_train_loader = None
    eval_val_loader = None

    num_train_samples = [11517, 251]
    num_val_samples = [2652, 129]

args["num_classes"] = num_classes

print_num_params(model)

#model = torch.load(f'{args["save_path"]}/init_model.pth')
model = model.to(device)

criterion = nn.CrossEntropyLoss()

with open(f'{args["save_path"]}/result.yaml','r') as f:
    args["result"]=yaml.load(f,Loader=yaml.UnsafeLoader)


dy = get_init_dy(args, num_train_samples)
ly = get_init_ly(args, num_train_samples)
w_train = get_train_w(args, num_train_samples)
w_val = get_val_w(args, num_val_samples)

print(f"w_train: {w_train}\nw_val: {w_val}")
print('ly', ly, '\n dy', dy)

if dataset == 'tvarit':
    args["epoch"]=500
    args["low_lr_schedule"]=[160,180]
    args["up_lr_schedule"]=[160,180]
    up_start_epoch = args["up_configs"]["start_epoch"]
    def warm_up_with_multistep_lr_low(epoch):
        return (epoch + 1) / args["low_lr_warmup"] \
            if epoch < args["low_lr_warmup"] \
            else 0.1 ** len([m for m in args["low_lr_schedule"] if m <= epoch])


    def warm_up_with_multistep_lr_up(epoch):
        return (epoch - up_start_epoch + 1) / args["up_lr_warmup"] \
            if epoch - up_start_epoch < args["up_lr_warmup"] \
            else 0.1 ** len([m for m in args["up_lr_schedule"] if m <= epoch])



    train_optimizer = optim.SGD(model.parameters(), lr=10**(-4), momentum=0.9, weight_decay=1e-4)
    val_optimizer = optim.SGD(params=[{'params': dy}, {'params': ly}, {'params': w_val}],
                              lr=args["up_lr"], momentum=0.9, weight_decay=1e-4)

    train_lr_scheduler = optim.lr_scheduler.LambdaLR(
        train_optimizer, lr_lambda=warm_up_with_multistep_lr_low)
    val_lr_scheduler = optim.lr_scheduler.LambdaLR(
        val_optimizer, lr_lambda=warm_up_with_multistep_lr_up)

if not os.path.exists(f'{args["save_path"]}/retrain'):
    os.makedirs(f'{args["save_path"]}/retrain')



torch.save(model, f'{args["save_path"]}/retrain/init_model.pth')
logfile = open(f'{args["save_path"]}/retrain/logs.txt', mode='w')
dy_log = open(f'{args["save_path"]}/retrain/dy.txt', mode='w')
ly_log = open(f'{args["save_path"]}/retrain/ly.txt', mode='w')
err_log = open(f'{args["save_path"]}/retrain/err.txt', mode='w')
with open(f'{args["save_path"]}/retrain/config.yaml', mode='w') as config_log:
    yaml.dump(args, config_log)

save_data = {"ly": [], "dy": [], "w_train": [],
             "train_err": [], "balanced_train_err": [], 
             "val_err": [], "balanced_val_err": [], "test_err": [], "balanced_test_err": [],
             "classwise_train_err": [], "classwise_val_err": [], "classwise_test_err": [] }
with open(f'{args["save_path"]}/retrain/result.yaml', mode='w') as log:
    yaml.dump(save_data, log)

for i in range(args["checkpoint"], args["epoch"]+1):
    if i % args["checkpoint_interval"] == 0:
        torch.save(model, f'{args["save_path"]}/retrain/epoch_{i}.pth')

    # if i % args["eval_interval"] == 0:
    #     text, loss, train_err, balanced_train_err, classwise_train_err = eval_epoch(eval_train_loader, model,
    #                                                            loss_adjust_cross_entropy, i, ' train_dataset', args,
    #                                                            params=[dy, ly, w_train])
    #     logfile.write(text+'\n')
    #     text, loss, val_err, balanced_val_err, classwise_val_err = eval_epoch(eval_val_loader, model,
    #                                                        cross_entropy, i, ' val_dataset', args, params=[dy, ly, w_val])
    #     logfile.write(text+'\n')

        text, loss, test_err, balanced_test_err, classwise_test_err = eval_epoch(test_loader, model,
                                                             cross_entropy, i, ' test_dataset', args, params=[dy, ly], encoder=Encoder_src)
        test_auc = evaluate_model_tvarit(test_loader, model, cross_entropy, i, ' test_dataset', args,
                                         encoder=Encoder_src)
        logfile.write(text+'\n')
    # save_data["train_err"].append(train_err)
    # save_data["balanced_train_err"].append(balanced_train_err)
    # save_data["classwise_train_err"].append(classwise_train_err)
    # save_data["val_err"].append(val_err)
    # save_data["balanced_val_err"].append(balanced_val_err)
    # save_data["classwise_val_err"].append(classwise_val_err)
    save_data["test_err"].append(test_err)
    save_data["balanced_test_err"].append(balanced_test_err)
    save_data["classwise_test_err"].append(classwise_test_err)

    save_data["dy"].append(dy.detach().cpu().numpy().tolist())
    save_data["ly"].append(ly.detach().cpu().numpy().tolist())
    save_data["w_train"].append(w_train.detach().cpu().numpy().tolist())

    with open(f'{args["save_path"]}/retrain/result.yaml', mode='w') as log:
        yaml.dump(save_data, log)

    print('ly', ly, '\n dy', dy, '\n')

    train_epoch(i, model, args,
                low_loader=train_loader, low_criterion=loss_adjust_cross_entropy,
                low_optimizer=train_optimizer, low_params=[dy, ly, w_train],
                up_loader=val_loader, up_optimizer=val_optimizer,
                up_criterion=cross_entropy, up_params=[dy, ly, w_val], encoder=Encoder_src)

    logfile.write(str(dy)+str(ly)+'\n\n')
    dy_log.write(f'{dy.detach().cpu().numpy()}\n')
    ly_log.write(f'{ly.detach().cpu().numpy()}\n')
    # err_log.write(f'{train_err} {val_err} {test_err}\n')
    logfile.flush()
    dy_log.flush()
    ly_log.flush()
    err_log.flush()
    train_lr_scheduler.step()
    val_lr_scheduler.step()

logfile.close()
dy_log.close()
ly_log.close()
err_log.close()
torch.save(model, f'{args["save_path"]}/retrain/final_model.pth')
test_auc = evaluate_model_tvarit(test_loader, model, cross_entropy, 500, ' test_dataset', args, encoder=Encoder_src)
train_auc = evaluate_model_tvarit(train_loader, model, cross_entropy, 500, ' train_dataset', args, encoder=Encoder_src)
val_auc = evaluate_model_tvarit(val_loader, model, cross_entropy, 500, ' val_dataset', args, encoder=Encoder_src)
