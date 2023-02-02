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
from sklearn.manifold import TSNE
import sklearn.metrics as sm
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from data_utils import *
from resnet import *
from raw_tvarit import *
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
    parser.add_argument('--num_meta', type=int, default=30,
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

def build_model(global_configs, meta_model=True):
    global encoder_checkpoint_path, args
    Encoder_src = Encoder_LSTM(n_features=global_configs['Source_no_of_features'], embedding_dim=global_configs['latent_dim'],N=global_configs['enc_layers'])
    Encoder_src.load_state_dict(torch.load(encoder_checkpoint_path))
    f1,f2,f3 = 256,128,64#args.def_f1,args.def_f2,args.def_f3
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
    
def plot_tsne_embeddings(data_loader, encoder, id_str):
    embed_list = None
    target_list = []
    for i, (input_var,targets,_,_) in enumerate(data_loader):
        embeds = encoder(input_var.cuda())
        embeds = embeds.cpu().detach().numpy()
        if embed_list is None:
            embed_list = embeds
        else:
            embed_list = np.concatenate((embed_list, embeds), axis = 0)
        targets = targets.cuda().view(-1)
        targets = targets.cpu().detach().numpy().tolist()
        target_list.extend(targets)
    
    #tsne = TSNE(n_components=3)
    #tsne_embeds = tsne.fit_transform(embed_list)
    targets_arr = np.array(target_list)
    targets_arr = targets_arr.reshape((len(target_list),1))
    data_embeds = np.concatenate((embed_list,targets_arr),axis=1)
    save_path = './' + id_str + '_embeds.npy'
    np.save(save_path,data_embeds)
    #tsne_good_list = []
    #tsne_bad_list = []
    #for i,j in enumerate(target_list):
  #      if j ==0:
   #         tsne_good_list.append(i)
    #    else:
     #       tsne_bad_list.append(i)

    #markers = np.array(["o" if i == 0 else "^" for i in target_list])
    #colors = np.array(["red" if i == 0 else "blue" for i in target_list])
    #labels = np.array(["good wheel" if i==0 else "bad wheel" for i in target_list])
#    good_label = "good_wheel"
 #   bad_label = "bad_wheel"
    #for embed_x, embed_y, color, m, lab in zip(tsne_embeds[:,0],tsne_embeds[:,1],colors,markers,labels):
  #  fig = plt.figure(figsize= (10,7))
   # ax = plt.axes(projection="3d")
  #  tsne_embeds_good = tsne_embeds[tsne_good_list]
  #  tsne_embeds_bad = tsne_embeds[tsne_bad_list]
  #  ax.scatter3D(tsne_embeds_good[:,0], tsne_embeds_good[:,1], tsne_embeds_good[:,2], color="red", marker="o", label=good_label)
   # ax.scatter3D(tsne_embeds_bad[:,0], tsne_embeds_bad[:,1], tsne_embeds_bad[:,2], color="blue", marker="^", label=bad_label)    
   # plt.legend()
   # plt.title("TSNE Plot for " + id_str + " Embeddings")
   # plt.savefig('./' + id_str + '_tsne.png')
   # plt.close()
    
def main():
    global args, best_prec1, model, optimizer_a, optimizer_c, vnet

    args, configs = parse_configs_and_args()
    # create model
    Encoder, _ = build_model(configs, meta_model=False)
    Encoder = Encoder.eval()
    cudnn.benchmark = True
    plot_tsne_embeddings(configs['Source_Train_Loader'], Encoder, 'Train')
    print("train done")
    plot_tsne_embeddings(configs['Source_Eval_Loader'], Encoder, 'Val')
    print("val done")
    plot_tsne_embeddings(configs['Source_Test_Loader'], Encoder, 'Test')
    print("test_done")
    
    

if __name__ == '__main__':
    main()




