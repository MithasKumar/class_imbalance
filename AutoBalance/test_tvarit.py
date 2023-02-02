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

from core.trainer import train_epoch, eval_epoch, evaluate_model_tvarit, evaluate_old_model_tvarit
from core.utils import loss_adjust_cross_entropy, cross_entropy,loss_adjust_cross_entropy_cdt
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
assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled


## Tvarit data imports

import yaml
from wheel_exp_main import get_models
from dataset.raw_tvarit import get_tvarit_raw


class YAML:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, item):
        return self.data[item]

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config',
                    default="./configs/Wheel/dyly_no_init/1.yaml", type=str)
parser.add_argument('--model_type', dest='model_type', default="encoder", type=str)
args = parser.parse_args()

model_type = args.model_type

with open(args.config, mode='r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
device = args["device"]
dataset = args["dataset"]

args['model_type'] = model_type

print('Dataset Name', dataset)

if dataset == 'Cifar10':
    num_classes = 10
    model = ResNet32(num_classes=num_classes)
    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_cifar10(
        train_size=args["train_size"], val_size=args["val_size"],
        balance_val=args["balance_val"], batch_size=args["low_batch_size"],
        train_rho=args["train_rho"],
        image_size=32, path=args["datapath"])
    Encoder_src = None
elif dataset == 'Cifar100':
    num_classes = 100
    model = ResNet32(num_classes=num_classes)
    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_cifar100(
        train_size=args["train_size"], val_size=args["val_size"],
        balance_val=args["balance_val"], batch_size=args["low_batch_size"],
        train_rho=args["train_rho"],
        image_size=32, path=args["datapath"])
elif dataset == 'ImageNet':
    num_classes = 1000
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=[0, 1])

    train_loader = ImageNetLTDataLoader(
        args["datapath"],
        training=True, batch_size=args["low_batch_size"])
    val_loader = train_loader.split_validation()
    test_loader = ImageNetLTDataLoader(
        args["datapath"],
        training=False, batch_size=args["low_batch_size"])

    num_train_samples, num_val_samples = train_loader.get_train_val_size()
    eval_train_loader = ImageNetLTDataLoader(
        args["datapath"],
        training=True, batch_size=512)
    eval_val_loader = eval_train_loader.split_validation()

elif dataset == 'INAT':
    num_classes = 8142
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=[0, 1])
    train_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                        is_train=True, split=1)
    num_train_samples = train_loader.get_class_size()
    train_loader = DataLoader(train_loader, batch_size=args["low_batch_size"],num_workers=6)
    val_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                      is_train=True, split=2)
    num_val_samples = val_loader.get_class_size()
    val_loader = DataLoader(val_loader, batch_size=args["low_batch_size"],num_workers=6)

    test_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/val2018.json',
                       is_train=False, split=0)
    test_loader.get_class_size()
    test_loader= DataLoader(test_loader,batch_size=512,num_workers=6)

    eval_train_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                        is_train=False, split=1)
    eval_train_loader = DataLoader(eval_train_loader, batch_size=512,num_workers=6)
    eval_val_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                      is_train=False, split=2)
    eval_val_loader = DataLoader(eval_val_loader, batch_size=512,num_workers=6)

elif dataset == 'tvarit':

    num_classes = 2
    config_file = './configs/Wheel/tvarit_wheel.yaml'


    # with open(r'/home/ece/Lakshya/Projects/prj_imb/AutoBalance/configs/Wheel/tvarit_wheel.yaml') as file:
    #     tvarit_args = yaml.load(file, Loader=yaml.FullLoader)

    with open(config_file, mode='r') as f:
        tvarit_args = YAML(yaml.load(f, Loader=yaml.FullLoader))

    print(tvarit_args)

    global_configs, Encoder_src, model = get_models(tvarit_args, model_type=args['model_type'])

    # model = nn.DataParallel(model, device_ids=[0, 1])
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



#import sys
#sys.exit()

print_num_params(model)


model = torch.load('/home/paperspace/Documents/repos/class_imbalance/AutoBalance/results/tvarit_wheel/New_Experiments/test_big_data/dyly_LA_init/1/retrain/final_model.pth')
    # model.load_state_dict(torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth'))
model = model.to(device)

print('AUROC for AutoBalance')
train_auc = evaluate_model_tvarit(train_loader, model, cross_entropy, 0, ' train_dataset', args, encoder=Encoder_src)
val_auc = evaluate_model_tvarit(val_loader, model, cross_entropy, 0, ' val_dataset', args, encoder=Encoder_src)
test_auc = evaluate_model_tvarit(test_loader, model, cross_entropy, 0, ' test_dataset', args, encoder=Encoder_src)


# print('AUROC for Old method')

# f1 = 256
# f2 = 128
# f3 = 64
#
# import dataset.train_utils.models as models
#
# model_old = models.MLP_model_old(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim'])
# model_old.load_state_dict(torch.load('/home/ece/Lakshya/Projects/prj_imb/Final_DIR_SITARAM/Final_DIR_SITARAM/Models/TEST_ID/Classifier.pt'))
# model_old = model_old.to(device)
#
# train_auc = evaluate_old_model_tvarit(train_loader, model_old, cross_entropy, 0, ' train_dataset', args, encoder=Encoder_src)
# val_auc = evaluate_old_model_tvarit(val_loader, model_old, cross_entropy, 0, ' val_dataset', args, encoder=Encoder_src)
# test_auc = evaluate_old_model_tvarit(test_loader, model_old, cross_entropy, 0, ' test_dataset', args, encoder=Encoder_src)
