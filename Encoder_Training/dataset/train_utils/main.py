# %%
def sitaram():
    return "Siiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiitaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaam"
# # print(sitaram())

# %%
# *********************** Importing Essential Libraries *************************
import os
import json
import math
import argparse
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import train_procedures
import train_utilities
import test_procedures
import utilis
import models
import data_processing
import inference

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device name :- "+str(device))

# ************** Initializing Argparser ********************
parser = argparse.ArgumentParser(description='Standard ML hyper-params')

# General Arguments (Optimizer(s) Related) --> 
parser.add_argument('--lr_src_enc_dec',help="Learning Rate for Source Encoder & Decoder (if any)",type=float,default=5*10**(-4))
parser.add_argument('--lr_tar_enc_dec',help="Learning Rate for Target Encoder & Decoder (if any)",type=float,default=5*10**(-4))
parser.add_argument('--wd_src_enc_dec',help="Weight Decay for Source Encoder & Decoder (if any)",type=float,default=0)
parser.add_argument('--beta1_src',help="Beta1 param for Source Encoder & Decoder (if any)",type=float,default=0.9) # Used only with Adam Optimizer
parser.add_argument('--beta2_src',help="Beta2 param for Source Encoder & Decoder (if any)",type=float,default=0.999) # Used only with Adam Optimizer
parser.add_argument('--beta1_tar',help="Beta1 param for Target Encoder & Decoder (if any)",type=float,default=0.9) # Used only with Adam Optimizer
parser.add_argument('--beta2_tar',help="Beta2 param for Target Encoder & Decoder (if any)",type=float,default=0.999) # Used only with Adam Optimizer
parser.add_argument('--lr_tar_gen',help="Learning Rate for Target Encoder as Generator",type=float,default=10**(-4))
parser.add_argument('--lr_disc',help="Learning Rate for Discriminator",type=float,default=10**(-4))
parser.add_argument('--lr_src_enc_cls',help="Learning Rate for Source Encoder while Learning from Classifier",type=float,default=10**(-4))
parser.add_argument('--beta1_disc',help="Beta1 param for Discriminator",type=float,default=0.5) # Used only with Adam Optimizer
parser.add_argument('--beta2_disc',help="Beta2 param for Discriminator",type=float,default=0.999) # Used only with Adam Optimizer
parser.add_argument('--beta1_cls',help="Beta1 param for Classifier",type=float,default=0.9) # Used only with Adam Optimizer
parser.add_argument('--beta2_cls',help="Beta2 param for Classifier",type=float,default=0.999) # Used only with Adam Optimizer
parser.add_argument('--beta1_gen',help="Beta1 param for Generator",type=float,default=0.5) # Used only with Adam Optimizer
parser.add_argument('--beta2_gen',help="Beta2 param for Generator",type=float,default=0.999) # Used only with Adam Optimizer
parser.add_argument('--lr_src_cls',help="Learning Rate for Source Classifier",type=float,default=10**(-4))
parser.add_argument('--wd_src_cls',help="Weight Decay for Source Classifier",type=float,default=0)
parser.add_argument('--lr_tar_cls',help="Learning Rate for Target Classifier",type=float,default=10**(-4)) # Used only tuning last layer classifier weights with Target labels
parser.add_argument('--wd_tar_cls',help="Weight Decay for Target Classifier",type=float,default=0) # Used only tuning last layer classifier weights with Target labels
parser.add_argument('--epochs',type=int,default=501,help="Number of Epochs")  
parser.add_argument('--target_recons_wait_epochs',type=int,default=0,help="Used only When Domain == Target, i.e., training target pipeline")
parser.add_argument('--target_tune_classifier_epochs',type=int,default=5,help="Used only When Domain == Target and a few Target labels are provided for fine-tuning classifier's last layer weights") 
parser.add_argument('--opt_gd',type=str,default='Adam',help="Optimizer Name for Generator-Discriminator Pair") 
parser.add_argument('--opt_ed_src',type=str,default='Adam',help="Optimizer Name for Source Encoder-Decoder Pair")
parser.add_argument('--opt_ed_tar',type=str,default='Adam',help="Optimizer Name for Target Encoder-Decoder Pair")
parser.add_argument('--opt_cls',type=str,default='Adam',help="Optimizer Name for Classifier")

# LR Scheduling Related Arguments -->
parser.add_argument('--schdl_opt_enc_src',type=str,help="Scheduler for Source Encoder",default=None) # list with info --> [eta_min,eta_max,ad_fac=Ascend/Descend,min_epoch=10,max_epoch=50]
parser.add_argument('--schdl_opt_enc_tar',type=str,help="Scheduler for Target Encoder",default=None)
parser.add_argument('--schdl_opt_dec_src',type=str,help="Scheduler for Source Decoder",default=None)
parser.add_argument('--schdl_opt_dec_tar',type=str,help="Scheduler for Target Decoder",default=None)
parser.add_argument('--schdl_opt_disc',type=str,help="Scheduler for Discriminator",default=None)
parser.add_argument('--schdl_opt_gen',type=str,help="Scheduler for Generator",default=None)
parser.add_argument('--schdl_opt_cls',type=str,help="Scheduler for Classifier",default=None)
parser.add_argument('--schdl_opt_cls_tar',type=str,help="Scheduler for Target Classifier",default=None) # Used only when tuning classifier weights for outer layer neuron added while classifier tuning on Target Labels
parser.add_argument('--schdl_opt_cls_enc',type=str,help="Scheduler for Source Encoder while getting Backprop error from Classifier",default=None)
parser.add_argument('--prnt_lr_updates',type=bool,help="To Print After Each Epoch Updated Learning Rates if using a Scheduler (Suggested: True)",default=True)

# Basic Hyperparameters -->
parser.add_argument('--bs',help="Batch Size",type=int,default=32)
parser.add_argument('--ld',help="Latent Dimension Size",type=int,default=64)
parser.add_argument('--tsql',type=int,help="Length used by Adaptive Pooling for making Batches of Raw Data",default=None)
parser.add_argument('--data_type',type=str,help="Which Data to Experiment with (Raw/HCF)",default='Raw') 
parser.add_argument('--train_tar_rat',type=str,default=None,help="Ratio of Target Training Data to be used")

# Loading Arguments -->
parser.add_argument('--load_id',type=str,default=None,help="JobID corresponding to trained Source Pipeline Models")
parser.add_argument('--load_id2',type=str,default=None,help="JobID corresponding to trained Target Pipeline Models")
parser.add_argument('--load_mn_src_enc',type=str,default='Encoder_src',help="Name of Source Encoder's state_dict.pt to be loaded")
parser.add_argument('--load_mn_tar_enc',type=str,default='Encoder_tar',help="Name of Target Encoder's state_dict.pt to be loaded")
parser.add_argument('--load_mn_src_dec',type=str,default='Decoder_src',help="Name of Source Decoder's state_dict.pt to be loaded")
parser.add_argument('--load_mn2',type=str,default='Classifier',help="Name of Classifier's state_dict.pt to be loaded")
parser.add_argument('--load_mn2_tar',type=str,default="Classifier",help="Name of Classifier's (from target pipeline) state_dict.pt to be loaded")
parser.add_argument('--load_pre_train_enc',type=str,default='All_layers',help="Load Source Encoder Weights as Target Encoder Weights before starting training the target pipeline") # options=["None","All_layers","Except_first_layer"]
parser.add_argument('--freeze_enc_tar_till',type=str,default=None,help="Number of epochs until which only Target Encoder's last layer is trained, Default: None which means train all the target encoder's weights from beginning")
parser.add_argument('--load_pre_train_dec',type=bool,default=False,help="Whether or not to load pre-trained weights from Source Decoder (if any) into Target Decoder (if any)")

# Utility Function(s) related Arguments -->
parser.add_argument('--tsne',type=str,default='Both',help="Domains for which t-SNE plots are required")
parser.add_argument('--threshold_selection_criterion',type=str,default='spec_sens_norm',help="How to pick the best threshold from the Eval Set Labels")
parser.add_argument('--threshold_penalty_weight',type=float,default=0.1,help="Weight of penalty imposed for picking a threshold with specificity-sensitivity values too off from each other")
parser.add_argument('--save_embeds',type=bool,default=False,help="To save the embeddings")
parser.add_argument('--tsne_labels',type=int,default=1,help="Plot t-SNEs for True labels(1)/Set-point params(2)/Both(0)") # 0 --> Both; 1 --> tru lbls only; 2 --> sp only
parser.add_argument('--data_norm_lvl',type=str,default='Local',help="Normaliza Data on Local/Global Level")
parser.add_argument('--data_norm_mode',type=str,default=None,help="Type of Normalization") # options=["Standard_Scalar","Min_Max"]
parser.add_argument('--impose_scalers',type=bool,default=False,help="To impose Normalization params calculated from Source Data on Target Data")
parser.add_argument('--temp_thresh',type=float,default=2001,help="Used with Raw Data for not considering temperature column values above temp_thresh while calculating Normalization params")
parser.add_argument('--upsample',type=str,default=None,help="Method to be used for upsampling HCF Data") # To be used while loading HCF Data # options = [SMOTE,SMOTEEN]
parser.add_argument('--prnt_processing_info',type=bool,default=False,help="To Print all the processing steps while processing HCF Data, Suggested: False for brevity") # To be used while loading HCF Data
parser.add_argument('--seed_upsampler',type=bool,default=False,help="To give upsampler same seed as used globally throughout the code, else it'll seed the sampler as 42") # To be used while loading HCF Data
parser.add_argument('--use_loader_sampler',type=bool,default=False,help="To use Imbalance Data Sampler for Raw data") # To be used while loading Raw Data
parser.add_argument('--want_probab_plots',type=bool,default=False,help="To plot probability plots")

# Adversarial Training Related -->
parser.add_argument('--recons_loss_func',type=str,default='MSE',help="Loss Function used for Reconstruction")
parser.add_argument('--gan_mode',type=str,default='WGAN',help="Type of GAN required") # options=["SNGAN","GAN","WGAN"]
parser.add_argument('--lpchz_mode',type=str,default='GP',help="Method used for imposing the Lipchitz Condition (Applicable only for WGANs)")
parser.add_argument('--gp_lambda',type=float,default=11,help="if lpchz_mode == GP, then gp_lambda denotes the weight of gradient penalty")
parser.add_argument('--opt_type',type=str,default='max_min',help="Type of Optimization for WGAN")
parser.add_argument('--f1d',type=int,default=512,help="Number of nuerons in the FIRST layer of Discriminator Architecture")
parser.add_argument('--f2d',type=int,default=256,help="Number of nuerons in the SECOND layer of Discriminator Architecture")
parser.add_argument('--f3d',type=int,default=128,help="Number of nuerons in the THIRD layer of Discriminator Architecture")
parser.add_argument('--critic_iters',type=int,default=-1,help="Number of Discriminator/Critic updates per Generator update, defaults to standard values for diff types of GANs")
parser.add_argument('--clp_l',type=float,default=-0.01,help="Lower Value for Clipping Weights (in WGAN + Weight Clipping)")
parser.add_argument('--clp_u',type=float,default=0.01,help="Upper Value for Clipping Weights (in WGAN + Weight Clipping)")
parser.add_argument('--enc_layers',type=int,default=3,help="Number of layers in the Encoder Architecture")
parser.add_argument('--dec_layers',type=int,default=None,help="Number of layers in the Decoder Architecture, if None then sets it equal to the number of encoder layers")
parser.add_argument('--enc_layers_info',type=str,default='[256,128]',help="Number of nuerons in each layer of Encoder Architecture, used with HCF Data since the Encoder Architecture's a MLP for HCF Data") # Used while loading HCF Data
parser.add_argument('--dec_layers_info',type=str,default=None,help="Number of nuerons in each layer of Decoder Architecture, used with HCF Data since the Decoder Architecture's a MLP for HCF Data, if set to None then it approximately mimics the encoder architecture in reverse") # Used while loading HCF Data
parser.add_argument('--gen_iters',type=int,default=1,help="Number of generator updates per Discriminator/Critic Update")
parser.add_argument('--adv_pen',type=float,default=None,help="Weight given to Adversarial Penalty")
parser.add_argument('--adv_pen_perc',type=float,default=50,help="Percentage of weight given to Adversarial Penalty, used when adv_pen isn't supplied explicitly")
parser.add_argument('--gan_sl',type=float,default=1,help="Smooth Label to used for more stable GAN training (Applicable only with OG GANs)")

# Classifier Related Arguments -->
parser.add_argument('--classifier_loss_func',type=str,default='WFL',help="Loss Function used for Classification")
parser.add_argument('--wfl_alpha',type=float,default=0.07, help="Alpha for WFL (if used as classifier_loss_func)")
parser.add_argument('--wfl_gamma',type=int,default=2, help="Gamma for WFL (if used as classifier_loss_func)")
parser.add_argument('--f1',type=str,default=None,help="Number of nuerons in first layer(s) of classifier(s) (Plural in case of Ensemble Classifier)")
parser.add_argument('--f2',type=str,default=None,help="Number of nuerons in second layer(s) of classifier(s) (Plural in case of Ensemble Classifier)")
parser.add_argument('--f3',type=str,default=None,help="Number of nuerons in third layer(s) of classifier(s) (Plural in case of Ensemble Classifier)")
parser.add_argument('--def_f1',type=int,default=256,help="Default of number of neurons in first layer(s) of the classifier(s) (Applicable in case --f1 is None)")
parser.add_argument('--def_f2',type=int,default=128,help="Default of number of neurons in second layer(s) of the classifier(s) (Applicable in case --f2 is None)")
parser.add_argument('--def_f3',type=int,default=64,help="Default of number of neurons in third layer(s) of the classifier(s) (Applicable in case --f3 is None)")
parser.add_argument('--ensemble',type=bool,default=False,help="True if classifier's an ensemble of MLPs")
parser.add_argument('--model_mode',type=str,default=None,help="Ensemble classifier's mode of making predictions") # options=["Majority_Vote","Prod_Probabs"]
parser.add_argument('--N',type=int,default=3,help="No of 3-layered MLP classifiers to be used in the Ensemble Classifier")
parser.add_argument('--lam',type=str,default='0.1',help="Threshold to be used on classifier outputs (A list if the classifier's an ensemble) (This threshold is to be provided if not training and just loading the classifier weights)") # Example --> [source_thresh,target_thresh] (just_infer and if there are 2 nuerons in classifier for separate domains)
parser.add_argument('--cls_pen',type=float,default=None,help="Weight given to Classification Penalty (if None => cls_pen set to 1)")
parser.add_argument('--classify_every_epoch',type=bool,default=True,help="Whether to infer with Classifier on target data after every epoch or not (Used while training only target encoder)")
parser.add_argument('--window_classifier_configs',type=str,default='[256,128]',help="If using Window Classifier as a Decoder, then pass number of neurons in each layer in the form of a list")

# Misc -->
parser.add_argument('--jobid',type=str,default="BIG_DATA_TRY1",help="A Unique JobID associated with each run of main.py")
parser.add_argument('--only_deterministic',type=bool,default=False,help="True to set pytorch's engine to only_deterministic mode (For reproducibility)")
parser.add_argument('--rand_seed',type=int,default=-1,help="If set to -1, randomly generates a seed and prints it")
parser.add_argument('--mn1_src',type=str,default='Encoder_src',help="Name of Source Encoder's state_dict.pt")
parser.add_argument('--mn1_tar',type=str,default='Encoder_tar',help="Name of Target Encoder's state_dict.pt")
parser.add_argument('--mn2_src',type=str,default='Decoder_src',help="Name of Source Decoder's state_dict.pt (if any)")
parser.add_argument('--mn2_tar',type=str,default='Decoder_tar',help="Name of Target Decoder's state_dict.pt (if any)")
parser.add_argument('--mn3',type=str,default='Discriminator',help="Name of Discriminator's state_dict.pt (Used only while training the target pipeline)")
parser.add_argument('--mn4',type=str,default='Classifier' ,help="Name of Classifier's state_dict.pt")

# Domain Adaptation Related -->
parser.add_argument('--train_mode',type=str,default='Source_pipeline') # options=["Source_pipeline","Target_pipeline"]
parser.add_argument('--recons',type=str,default="Win_Zeros_4",help="Type of Decoder to be used (Window Classifier or LSTM based Reconstructor)") # if using window classification --> "Win_windowMode_windowNum" where "windowMode" can be "Zeros/Noise" and "windowNum" is an int denoting number of classes; if using LSTM-based --> "LSTM"
parser.add_argument('--tune_tar_with_cls',type=bool,default=False,help="To use Target labels for fine-tuning Classifier weights")

args = parser.parse_args()
print(vars(args))

# %%
# *********************** Standard ML Hyper-parameters *************************
learning_rate1 = args.lr_src_enc_dec
learning_rate1_tar = args.lr_tar_enc_dec
learning_rate1_G = args.lr_tar_gen
learning_rate1_D = args.lr_disc
learning_rate1_cls = args.lr_src_enc_cls
weight_dr1 = args.wd_src_enc_dec
learning_rate2 = args.lr_src_cls
weight_dr2 = args.wd_src_cls
learning_rate2_tar = args.lr_tar_cls
weight_dr2_tar = args.wd_tar_cls
lr_configs = {}
lr_configs['optimizer_classifier'] = learning_rate2
lr_configs['optimizer_classifier_tar'] = learning_rate2_tar
lr_configs['optimizer_encoder_src'] = learning_rate1
lr_configs['optimizer_decoder_src'] = learning_rate1
lr_configs['optimizer_encoder_tar'] = learning_rate1_tar
lr_configs['optimizer_decoder_tar'] = learning_rate1_tar
lr_configs['optimizer_generator'] = learning_rate1_G
lr_configs['optimizer_critic'] = learning_rate1_D
lr_configs['optimizer_encoder_classifier'] = learning_rate1_cls

# t-SNE related variables -->
to_plt_tsne=args.tsne
if to_plt_tsne=="Both" and args.train_mode=="Source_pipeline":
    to_plt_tsne="Source"
want_sp = want_lbls = True
if args.tsne_labels==1:
    want_sp=False
if args.tsne_labels==2:
    want_lbls=False

just_infer = (args.load_id is not None) and (args.load_id2 is not None) # True if main.py is ran just for inferring 
if just_infer:
    args.train_mode=None
    recons_domain=None

# Encoder and Decoder Architecture Related Arguments -->
Enc_Lyrs = args.enc_layers
if args.dec_layers is None:
    Dec_Lyrs = Enc_Lyrs
else:
    Dec_Lyrs = args.dec_layers

# Decoder related Variables -->
recons_window=args.recons.split("_")[0]=="Win"
if not just_infer:
    recons_domain=args.train_mode.split("_")[0]

# **************** Some Relevant Paths **************************
root_path = "./" # CHANGE THIS TO Appropriate ROOT PATH
data_path = os.path.join(root_path,"Datasets")
model_path = os.path.join(root_path,"Models")
save_model_path = os.path.join(model_path,args.jobid)
save_plot_path = os.path.join(root_path,"Plots",args.jobid)
exist_ok_val=True
# if "TEST" in args.jobid:
    # exist_ok_val=True
os.makedirs(save_plot_path,exist_ok=exist_ok_val)
if not just_infer:
    os.makedirs(save_model_path,exist_ok=exist_ok_val)
if args.load_id is not None:
    load_src_enc_model_path = os.path.join(model_path,args.load_id,args.load_mn_src_enc+".pt")
    load_src_dec_model_path = os.path.join(model_path,args.load_id,args.load_mn_src_dec+".pt")
    load_model2_path = os.path.join(model_path,args.load_id,args.load_mn2+".pt")
USING_TARGET_CLS=False
if args.load_id2 is not None:
    assert args.load_id is not None, "Provide a Valid Load ID (for loading Source Models) first before providing Load ID2."
    load_tar_enc_model_path = os.path.join(model_path,args.load_id2,args.load_mn_tar_enc+".pt")
    if os.path.exists(os.path.join(model_path,args.load_id2,args.load_mn2_tar+".pt")):
        load_model2_path = os.path.join(model_path,args.load_id2,args.load_mn2_tar+".pt")
        USING_TARGET_CLS=True

# %%
# ************** Initializing Global Configs Dict ********************
global_configs = {}
global_configs['Source_configs_path'] = os.path.join(data_path,'cza_modelling_data_config.json')
global_configs['Target_configs_path'] = os.path.join(data_path,'da_target_configs.json')
global_configs['Source_train_data_path'] = os.path.join(data_path,'cza_train.feather')
global_configs['Source_eval_data_path'] = os.path.join(data_path,'cza_test.feather')
global_configs['Source_test_data_path'] = os.path.join(data_path,'cza_test.feather')
global_configs['Target_train_data_path'] = os.path.join(data_path,'da_target_train.feather')
global_configs['Target_eval_data_path'] = os.path.join(data_path,'da_target_eval.feather')
global_configs['Target_test_data_path'] = os.path.join(data_path ,'da_target_test.feather')
if args.data_type=='HCF':
    global_configs['data_path'] = data_path 
    # global_configs['domain'] = args.train_mode.split("_")[0]
    global_configs['upsample'] = args.upsample
    global_configs['seed_upsample'] = args.seed_upsampler
    global_configs['prnt_processing_info'] = args.prnt_processing_info
    global_configs['Source_no_of_features']=599
global_configs['batch_size'] = args.bs
global_configs['latent_dim'] = args.ld
if args.data_type=='Raw':
    global_configs['use_loader_sampler'] = args.use_loader_sampler
global_configs['to_save_embeds'] = args.save_embeds
if global_configs['to_save_embeds']:
    save_embeds_path = os.path.join(save_plot_path,'Embeddings')
    os.makedirs(save_embeds_path,exist_ok=exist_ok_val)
    global_configs['save_train_embeds_path'] = os.path.join(save_embeds_path,'both_domains_train_embeds.csv')
    global_configs['save_test_embeds_path'] = os.path.join(save_embeds_path,'both_domains_test_embeds.csv')
if args.tune_tar_with_cls or recons_domain=="Source":
    global_configs['save_classifier_path'] = os.path.join(save_model_path,args.mn4 +'.pt')
    global_configs['source_plot_path'] = os.path.join(save_plot_path,'Source_Plots')
    global_configs['target_plot_path']  = os.path.join(save_plot_path,'Target_Plots')
    os.makedirs(global_configs['source_plot_path'],exist_ok=exist_ok_val)
    os.makedirs(global_configs['target_plot_path'],exist_ok=exist_ok_val)

if not just_infer:
    global_configs['epochs'] = args.epochs
    if args.train_mode.split("_")[0]=="Source":
        global_configs['save_encoder_src_path'] = os.path.join(save_model_path,args.mn1_src +'.pt')
        global_configs['save_decoder_src_path'] = os.path.join(save_model_path,args.mn2_src +'.pt')
    elif args.train_mode.split("_")[0]=="Target":
        global_configs['save_encoder_tar_path'] = os.path.join(save_model_path,args.mn1_tar +'.pt')
        global_configs['save_decoder_tar_path'] = os.path.join(save_model_path,args.mn2_tar +'.pt')
        global_configs['save_discriminator_path'] = os.path.join(save_model_path,args.mn3 +'.pt')

if args.critic_iters!=-1:
    global_configs['critic_iters'] = args.critic_iters
else:
    if args.gan_mode in ['GAN','SNGAN']:
        global_configs['critic_iters'] = 1
    elif args.gan_mode=='WGAN':
        global_configs['critic_iters'] = 5
global_configs['want_sp'] = want_sp
global_configs['want_lbls'] = want_lbls
global_configs['unfreeze_first_lyr_only'] = False

if recons_domain=="Target":
    global_configs['tar_epochs_wait'] = args.target_recons_wait_epochs
    if args.tune_tar_with_cls:
        global_configs['cls_tar_epochs'] = args.target_tune_classifier_epochs
    if args.freeze_enc_tar_till is not None:
        global_configs['unfreeze_first_lyr_only'] = True
        global_configs['unfreeze_epochs'] = int(args.freeze_enc_tar_till)
            
global_configs['wgan_train_mode'] = args.lpchz_mode
global_configs['gp_penalty'] = args.gp_lambda
if args.data_type=='Raw':
    global_configs['adap_len'] = args.tsql
    if args.tsql is None:
        global_configs['adap_len'] = 301
global_configs['min_max'] = args.opt_type=='min_max'
if args.adv_pen is None:
    global_configs['adv_pen'] = args.adv_pen_perc/100
    global_configs['cls_pen'] = 1-global_configs['adv_pen']
else:
    global_configs['adv_pen'] = args.adv_pen
    if args.cls_pen is not None:
        global_configs['cls_pen'] = args.cls_pen
    else:
        global_configs['cls_pen'] = 1
global_configs['is_ensemble'] = args.ensemble
global_configs['ensemble_models'] = args.N
global_configs['ensemble_mode'] = args.model_mode
global_configs['recons_loss_func'] = args.recons_loss_func
global_configs['clip_upper'] = args.clp_u
global_configs['clip_lower'] = args.clp_l

global_configs['classifier_loss_func'] = args.classifier_loss_func
global_configs['wfl_alpha'] = args.wfl_alpha
global_configs['wfl_gamma'] = args.wfl_gamma
global_configs['thresh_selection'] = args.threshold_selection_criterion
global_configs['thresh_pen_weight'] = args.threshold_penalty_weight
if global_configs['thresh_selection']=='spec_sens':
    global_configs['thresh_pen_weight'] = 0
print("")
print("Penalty used for optimal threshold selection = "+str(global_configs['thresh_pen_weight']))
print("")
    
if args.classify_every_epoch and recons_domain=="Target":
    global_configs['target_plot_path']  = os.path.join(save_plot_path,'Target_Plots')
    os.makedirs(global_configs['target_plot_path'],exist_ok=exist_ok_val)
global_configs['save_plot_path'] = save_plot_path
global_configs['to_plt_tsne'] = to_plt_tsne
if to_plt_tsne!='None' and global_configs['want_sp']:
    global_configs['save_src_tsne_plots_path'] = os.path.join(save_plot_path,'t-SNE_Plots','Source')
    global_configs['save_tar_tsne_plots_path'] = os.path.join(save_plot_path,'t-SNE_Plots','Target')
    os.makedirs(global_configs['save_src_tsne_plots_path'],exist_ok=exist_ok_val)
    os.makedirs(global_configs['save_tar_tsne_plots_path'],exist_ok=exist_ok_val)
global_configs['only_deterministic'] = args.only_deterministic
if args.rand_seed!=-1:
    global_configs['rand_seed'] = args.rand_seed
else:
    global_configs['rand_seed'] = random.randint(0,(2**32)-1)
    print("Using Random Seed ==> "+str(global_configs['rand_seed']))

global_configs['gan_mode'] = args.gan_mode
if args.gan_mode=='SNGAN':
    global_configs['gan_mode'] = 'GAN'
if global_configs['gan_mode']=='WGAN':
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    global_configs['one'] = one
    global_configs['mone'] = mone
elif global_configs['gan_mode']=='GAN':
    global_configs['gan_smooth_label'] = args.gan_sl
global_configs['gen_iters'] = args.gen_iters

global_configs['classify_every_epoch'] = args.classify_every_epoch
global_configs['train_cls_tar'] = args.tune_tar_with_cls
if global_configs['train_cls_tar']:
    global_configs['classify_every_epoch']=True
    global_configs['Target_train_lbls_data_path'] = os.path.join(data_path,'da_target_train_lbls.feather')
    global_configs['Target_eval_lbls_data_path'] = os.path.join(data_path,'da_target_eval_lbls.feather')

if args.data_type=='Raw':
    global_configs['data_norm_lvl'] = args.data_norm_lvl
global_configs['data_norm_mode'] = args.data_norm_mode
if global_configs['data_norm_mode'] is None:
    if args.data_type=='Raw':
        global_configs['data_norm_mode']='Min_Max'
    elif args.data_type=='HCF':
        global_configs['data_norm_mode']='Standard_Scalar'
global_configs['impose_scalers'] = args.impose_scalers
if args.data_type=='Raw':
    global_configs['temp_thresh'] = args.temp_thresh
global_configs['prnt_lr_updates'] = args.prnt_lr_updates
global_configs['recons_window'] = recons_window
if args.recons != "None":
    global_configs['recons_domain'] = recons_domain

global_configs['want_probab_plots'] = args.want_probab_plots
if global_configs['want_probab_plots']:
    save_probab_plts_path = os.path.join(save_plot_path,'Probab_Plots')
    os.makedirs(save_probab_plts_path,exist_ok=exist_ok_val)
    global_configs['save_probab_plots_path'] = save_probab_plts_path
if recons_window:
    global_configs["win_recons_info"] = args.recons
    global_configs['win_cls_configs'] = args.window_classifier_configs
    global_configs['cls_lyrs'] = Dec_Lyrs
if args.data_type=='HCF':
    dec_sig = global_configs['data_norm_mode']=='Min_Max'
elif args.data_type=='Raw':
    dec_sig = global_configs['data_norm_mode']=='Min_Max' and global_configs['data_norm_lvl']=='Local'

if args.data_type=="Raw":
    global_configs["subset_dict"] = utilis.make_subset_dict(string=args.train_tar_rat)
opt_lst=[]
# print(sitaram())

# %%
"""
# Making Data Loaders --->
"""

# %%
# ************* Random Seed to all the devices ****************
torch.backends.cudnn.deterministic = global_configs['only_deterministic']
torch.manual_seed(global_configs['rand_seed'])
np.random.seed(global_configs['rand_seed'])
random.seed(global_configs['rand_seed'])
train_procedures.rand_seed_devices(configs=global_configs)
train_utilities.rand_seed_devices(configs=global_configs)
test_procedures.rand_seed_devices(configs=global_configs)
utilis.rand_seed_devices(configs=global_configs)
inference.rand_seed_devices(configs=global_configs)
data_processing.rand_seed_devices(configs=global_configs)
models.rand_seed_devices(configs=global_configs)
# print(sitaram())

# %%
if args.data_type=='Raw':
    global_configs = data_processing.get_raw_data_df(configs=global_configs) # Adding Raw Data's Dataframes
    data_processing.print_raw_data_df_info(configs=global_configs) # Print Dataset Info
    global_configs = data_processing.get_raw_data_to_loaders(configs=global_configs)
elif args.data_type=='HCF':
    global_configs = data_processing.get_hcf_data_to_loaders(configs=global_configs)
else:
    raise NotImplementedError("Data Type can't be recognized :(")

# %%
"""
# Init Models, Optimizers & Criterions -->
"""

if just_infer or (recons_domain=="Target"):
    # %%
    def make_flt_list(str_of_lst):
        res=[]
        temp_lst = str_of_lst.split(',')
        i=0
        while i<len(temp_lst):
            if i==0:
                if temp_lst[i][0]=='[':
                    res.append(float(temp_lst[i][1:]))
                else:
                    res.append(float(temp_lst[i]))
            elif i==len(temp_lst)-1:
                if temp_lst[i][-1]==']':
                    res.append(float(temp_lst[i][:-1]))
                else:
                    res.append(float(temp_lst[i]))
            else:
                res.append(float(temp_lst[i]))
            i+=1

        return res

    # print(sitaram())

    # %%
    if not global_configs['is_ensemble']:
        if USING_TARGET_CLS:
            [lamda,global_configs["best_threshold_tar"]] = args.lam[1:-1].split(",")
            lamda = float(lamda)
            global_configs["best_threshold_tar"]=float(global_configs["best_threshold_tar"])
        else:
            lamda = float(args.lam)
    else:
        if global_configs['ensemble_mode']=="Majority_Vote":
            lamda = make_flt_list(args.lam)
        elif global_configs['ensemble_mode']=="Prod_Probabs":
            lamda = float(args.lam)
    global_configs['best_threshold'] = lamda
    # print(sitaram())

if global_configs['is_ensemble']:
    # %%
    def make_neurons_list(f_str,idx):
        res = []
        if idx==1:
            def_val = args.def_f1
        elif idx==2:
            def_val = args.def_f2
        elif idx==3:
            def_val = args.def_f3
        if f_str is None:
            for i in range(global_configs['ensemble_models']):
                res.append(def_val)
        else:
            nuers_lst = f_str.split(',')
            for ele in nuers_lst:
                res.append(int(ele))
        return res
    # print(sitaram())

    # %%
    F1,F2,F3 = make_neurons_list(f_str=args.f1,idx=1),make_neurons_list(f_str=args.f2,idx=2),make_neurons_list(f_str=args.f3,idx=3)
    # print(sitaram())

# %%
# Initializing the classifier and its optimizer --->
f1,f2,f3 = args.def_f1,args.def_f2,args.def_f3
if just_infer:
    if USING_TARGET_CLS:
        Classifier = models.classifier_combined(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim']).to(device)
    else:
        if global_configs['is_ensemble']:
            Classifier = models.MLP_ensemble(F1=F1,F2=F2,F3=F3,ld=global_configs['latent_dim']).to(device)
        else:
            Classifier = models.MLP_model(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim']).to(device)
    Classifier.load_state_dict(torch.load(load_model2_path,map_location=device))
    Classifier.eval()
else:
    if global_configs['is_ensemble']:
        Classifier = models.MLP_ensemble(F1=F1,F2=F2,F3=F3,ld=global_configs['latent_dim']).to(device)
    else:
        Classifier = models.MLP_model(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim']).to(device)
    if args.train_mode=="Source_pipeline":
        global_configs['optimizer_classifier'] = torch.optim.Adam(Classifier.parameters(), lr=learning_rate2, weight_decay=weight_dr2)
        utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_classifier',str_lst=args.schdl_opt_cls,lr_info=lr_configs)
        opt_lst.append('optimizer_classifier')
    elif args.train_mode=="Target_pipeline":
        Classifier.load_state_dict(torch.load(load_model2_path,map_location=device))
        Classifier.eval()
        if args.tune_tar_with_cls:
            Classifier_tar = models.classifier_combined(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim']).to(device)
            Classifier,global_configs = utilis.load_pretrain_cls_tar(configs=global_configs,Cls_tar=Classifier_tar,Cls_src=Classifier,lr_sr=learning_rate2_tar,wd_sr=weight_dr2_tar)
            utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_classifier_tar',str_lst=args.schdl_opt_cls_tar,lr_info=lr_configs)
            opt_lst.append('optimizer_classifier_tar')
# print(sitaram())

# %%
# Initializing the encoder, decoder and discriminator and their optimizer(s) --->
if args.data_type=='HCF':
    F_encoder = utilis.make_neurons_list_wc(str_lst=args.enc_layers_info,N=Enc_Lyrs) # wc = window classifier
    if args.dec_layers_info is None:
        F_decoder = utilis.make_neurons_list_wc(str_lst=args.enc_layers_info,N=Dec_Lyrs)
    else:
        F_decoder = utilis.make_neurons_list_wc(str_lst=args.dec_layers_info,N=Dec_Lyrs)

if args.data_type=='Raw':
    Encoder_src = models.Encoder_LSTM(n_features=global_configs['Source_no_of_features'], embedding_dim=global_configs['latent_dim'],N=Enc_Lyrs).to(device)
    if args.train_mode=="Target_pipeline" or just_infer:
        Encoder_tar = models.Encoder_LSTM(n_features=global_configs['Target_no_of_features'], embedding_dim=global_configs['latent_dim'],N=Enc_Lyrs).to(device)
elif args.data_type=='HCF':
    Encoder_src = models.Encoder_MLP(nf=global_configs['Source_no_of_features'],N=Enc_Lyrs,F=F_encoder,ld=global_configs['latent_dim']).to(device)
    if args.train_mode=="Target_pipeline" or just_infer:
        Encoder_tar = models.Encoder_MLP(nf=global_configs['Target_no_of_features'],N=Enc_Lyrs,F=F_encoder,ld=global_configs['latent_dim']).to(device)
else:
    raise NotImplementedError("Enter Valid Data Type")

if not just_infer:

    if args.train_mode=="Source_pipeline":
        global_configs['optimizer_encoder_classifier'] = utilis.get_optimizer(model=Encoder_src, opt_name=args.opt_cls,lr=learning_rate1_cls,wd=0,b1=args.beta1_cls,b2=args.beta2_cls)
        utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_encoder_classifier',str_lst=args.schdl_opt_cls_enc,lr_info=lr_configs)
        opt_lst.append('optimizer_encoder_classifier')

    if args.recons!="None":
        if args.train_mode=="Source_pipeline":
            if recons_window:
                Decoder = models.MLP_model_multi_outs(N=global_configs['cls_lyrs'],nuerons_info=global_configs['window_classifer_info'],ld=global_configs['latent_dim']).to(device)
            else:
                if args.data_type=='Raw':
                    Decoder = models.Decoder_LSTM(input_dim=global_configs['latent_dim'], n_features=global_configs['Source_no_of_features'],N=Dec_Lyrs,tseq_len=global_configs['adap_len'],apply_sig=dec_sig).to(device)
                elif args.data_type=='HCF':
                    Decoder = models.Decoder_MLP(nf=global_configs['Source_no_of_features'],N=Dec_Lyrs,F=F_decoder,ld=global_configs['latent_dim'],rev_lst=args.dec_layers_info is None,apply_sig=dec_sig).to(device)  
                else:
                    raise NotImplementedError("Enter Valid Data Type")    
        elif args.train_mode=="Target_pipeline":
            if recons_window:
                Decoder = models.MLP_model_multi_outs(N=global_configs['cls_lyrs'],nuerons_info=global_configs['window_classifer_info'],ld=global_configs['latent_dim']).to(device)
            else:
                if args.data_type=='Raw':
                    Decoder = models.Decoder_LSTM(input_dim=global_configs['latent_dim'], n_features=global_configs['Target_no_of_features'],N=Dec_Lyrs,tseq_len=global_configs['adap_len'],apply_sig=dec_sig).to(device)
                elif args.data_type=='HCF':
                    Decoder = models.Decoder_MLP(nf=global_configs['Target_no_of_features'],N=Dec_Lyrs,F=F_decoder,ld=global_configs['latent_dim'],rev_lst=args.dec_layers_info is None,apply_sig=dec_sig).to(device)
                else:
                    raise NotImplementedError("Enter Valid Data Type")  
        else:
            raise NotImplementedError("Enter Valid Train Mode")
    else:
        Decoder=None        

    if args.train_mode=="Target_pipeline":
        Encoder_src.load_state_dict(torch.load(load_src_enc_model_path,map_location=device))
        if args.gan_mode=='SNGAN':
            Discriminator = models.SN_Discriminator(ld=global_configs['latent_dim'],f1d=args.f1d,f2d=args.f2d,f3d=args.f3d).to(device)
        else:
            Discriminator = models.Discriminator(ld=global_configs['latent_dim'],f1d=args.f1d,f2d=args.f2d,f3d=args.f3d,is_wgan=global_configs['gan_mode']=='WGAN').to(device)

        param_lst_tar_enc=None
        if args.load_pre_train_enc!="None":
            Encoder_tar,param_lst_tar_enc = utilis.load_pretrain_enc_weights(Encoder=Encoder_tar,Encoder_pt=Encoder_src,load_first_lyr=args.load_pre_train_enc=='All_layers',unfreeze_new_only=global_configs['unfreeze_first_lyr_only'])

    if args.train_mode=="Source_pipeline":
        global_configs['optimizer_encoder_src'] = utilis.get_optimizer(model=Encoder_src, opt_name=args.opt_ed_src,lr=learning_rate1,wd=weight_dr1,b1=args.beta1_src,b2=args.beta2_src)
        utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_encoder_src',str_lst=args.schdl_opt_enc_src,lr_info=lr_configs)
        opt_lst.append('optimizer_encoder_src')
    elif args.train_mode=="Target_pipeline":
        global_configs['optimizer_generator'] = utilis.get_optimizer(model=Encoder_tar, opt_name=args.opt_gd,lr=learning_rate1_G,wd=0,b1=args.beta1_gen,b2=args.beta2_gen,params_lst=param_lst_tar_enc)
        utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_generator',str_lst=args.schdl_opt_gen,lr_info=lr_configs)
        opt_lst.append('optimizer_generator')
        global_configs['optimizer_critic'] = utilis.get_optimizer(model=Discriminator, opt_name=args.opt_gd,lr=learning_rate1_D,wd=0,b1=args.beta1_disc,b2=args.beta2_disc)
        utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_critic',str_lst=args.schdl_opt_disc,lr_info=lr_configs)
        opt_lst.append('optimizer_critic')
    else:
        raise NotImplementedError("Enter Valid Train Mode")

    if Decoder is not None:
        if recons_domain=="Source":
            global_configs['optimizer_decoder_src'] = utilis.get_optimizer(model=Decoder, opt_name=args.opt_ed_src,lr=learning_rate1,wd=weight_dr1,b1=args.beta1_src,b2=args.beta2_src)
            utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_decoder_src',str_lst=args.schdl_opt_dec_src,lr_info=lr_configs)
            opt_lst.append('optimizer_decoder_src')
        elif recons_domain=="Target":
            global_configs['optimizer_encoder_tar'] = utilis.get_optimizer(model=Encoder_tar, opt_name=args.opt_ed_tar,lr=learning_rate1_tar,wd=0,b1=args.beta1_tar,b2=args.beta2_tar,params_lst=param_lst_tar_enc)
            utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_encoder_tar',str_lst=args.schdl_opt_enc_tar,lr_info=lr_configs)
            opt_lst.append('optimizer_encoder_tar')
            global_configs['optimizer_decoder_tar'] = utilis.get_optimizer(model=Decoder, opt_name=args.opt_ed_tar,lr=learning_rate1_tar,wd=0,b1=args.beta1_tar,b2=args.beta2_tar)
            utilis.update_scheduler(configs=global_configs,opt_nm='optimizer_decoder_tar',str_lst=args.schdl_opt_dec_tar,lr_info=lr_configs)
            opt_lst.append('optimizer_decoder_tar')
            if args.load_pre_train_dec:
                Decoder.load_state_dict(torch.load(load_src_dec_model_path,map_location=device))
        else:
            raise NotImplementedError("Reconstruction Domain undecipherable")

else:
    Encoder_src.load_state_dict(torch.load(load_src_enc_model_path,map_location=device))
    Encoder_tar.load_state_dict(torch.load(load_tar_enc_model_path,map_location=device))
    
global_configs['opt_lst'] = opt_lst
print("Optimizer Name(s) used for training (an empty list means you running the main.py in just_infer mode) => "+str(global_configs['opt_lst']))
print("")
# print(sitaram())

# %%
global_configs = utilis.add_criterions(configs=global_configs) # Adding Criterions to global configs dict
# print(sitaram())

# %%
"""
# Training Models SitaRam -->
"""

if not just_infer:
    if args.train_mode=="Source_pipeline":
        Encoder_src,Decoder,Classifier,global_configs['best_threshold'] = train_procedures.train_source_pipeline(Encoder_src=Encoder_src,Decoder_src=Decoder,Classifier=Classifier,configs=global_configs)
    elif args.train_mode=="Target_pipeline":
        Encoder_tar,Decoder,Discriminator,Classifier,thresh_tar = train_procedures.train_target_pipeline(Encoder_src=Encoder_src,Encoder_tar=Encoder_tar,Decoder_tar=Decoder,Classifier=Classifier,Discriminator=Discriminator,configs=global_configs)
        if thresh_tar is not None:
            global_configs['best_threshold_tar']=thresh_tar
    else:
        raise NotImplementedError("Train Mode not supported :(")
# print(sitaram())

# %%
if just_infer:
    Encoders = [Encoder_src,Encoder_tar]
else:
    if args.train_mode=="Source_pipeline":
        Encoders = [Encoder_src,None]
    elif args.train_mode=="Target_pipeline":
        Encoders = [Encoder_src,Encoder_tar]
    else:
        raise NotImplementedError("Train Mode not supported :(")

if args.data_type=='Raw':
    global_configs = data_processing.get_raw_data_to_embed_loaders(configs=global_configs,Encoder=Encoders) # For classification (Resetting the Data Loaders Accordingly)
elif args.data_type=='HCF':
    global_configs = data_processing.get_hcf_data_to_embed_loaders(configs=global_configs,Encoder=Encoders) # For classification (Resetting the Data Loaders Accordingly)
else:
    raise NotImplementedError("Data type not supported :(")
# print(sitaram())
# print(sitaram())

# %%
"""
# Plotting t-SNE -->
"""

# %%
if global_configs['to_plt_tsne']!='None':
    utilis.make_tsne(configs=global_configs)
# print(sitaram())

# %%
"""
# Inference on Classifier -->
"""

# %%
print("Inference Results -->")
inference.infer_classifier(Classifier=Classifier,configs=global_configs)
# print(sitaram())
# print(sitaram())