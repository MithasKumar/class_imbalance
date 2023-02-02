from dataset.raw_tvarit import get_tvarit_raw
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

import dataset.train_utils.train_procedures as train_procedures
import dataset.train_utils.train_utilities as train_utilities
import dataset.train_utils.test_procedures as test_procedures
import dataset.train_utils.utilis as utilis
import dataset.train_utils.models as models
import dataset.train_utils.data_processing as data_processing
import dataset.train_utils.inference as inference

import yaml

#
# # ************** Initializing Argparser ********************
# parser = argparse.ArgumentParser(description='Standard ML hyper-params')
#
# # General Arguments (Optimizer(s) Related) -->
# parser.add_argument('--lr_src_enc_dec', help="Learning Rate for Source Encoder & Decoder (if any)", type=float,
#                     default=5 * 10 ** (-4))
# parser.add_argument('--lr_tar_enc_dec', help="Learning Rate for Target Encoder & Decoder (if any)", type=float,
#                     default=5 * 10 ** (-4))
# parser.add_argument('--wd_src_enc_dec', help="Weight Decay for Source Encoder & Decoder (if any)", type=float,
#                     default=0)
# parser.add_argument('--beta1_src', help="Beta1 param for Source Encoder & Decoder (if any)", type=float,
#                     default=0.9)  # Used only with Adam Optimizer
# parser.add_argument('--beta2_src', help="Beta2 param for Source Encoder & Decoder (if any)", type=float,
#                     default=0.999)  # Used only with Adam Optimizer
# parser.add_argument('--beta1_tar', help="Beta1 param for Target Encoder & Decoder (if any)", type=float,
#                     default=0.9)  # Used only with Adam Optimizer
# parser.add_argument('--beta2_tar', help="Beta2 param for Target Encoder & Decoder (if any)", type=float,
#                     default=0.999)  # Used only with Adam Optimizer
# parser.add_argument('--lr_tar_gen', help="Learning Rate for Target Encoder as Generator", type=float,
#                     default=10 ** (-4))
# parser.add_argument('--lr_disc', help="Learning Rate for Discriminator", type=float, default=10 ** (-4))
# parser.add_argument('--lr_src_enc_cls', help="Learning Rate for Source Encoder while Learning from Classifier",
#                     type=float, default=10 ** (-4))
# parser.add_argument('--beta1_disc', help="Beta1 param for Discriminator", type=float,
#                     default=0.5)  # Used only with Adam Optimizer
# parser.add_argument('--beta2_disc', help="Beta2 param for Discriminator", type=float,
#                     default=0.999)  # Used only with Adam Optimizer
# parser.add_argument('--beta1_cls', help="Beta1 param for Classifier", type=float,
#                     default=0.9)  # Used only with Adam Optimizer
# parser.add_argument('--beta2_cls', help="Beta2 param for Classifier", type=float,
#                     default=0.999)  # Used only with Adam Optimizer
# parser.add_argument('--beta1_gen', help="Beta1 param for Generator", type=float,
#                     default=0.5)  # Used only with Adam Optimizer
# parser.add_argument('--beta2_gen', help="Beta2 param for Generator", type=float,
#                     default=0.999)  # Used only with Adam Optimizer
# parser.add_argument('--lr_src_cls', help="Learning Rate for Source Classifier", type=float, default=10 ** (-4))
# parser.add_argument('--wd_src_cls', help="Weight Decay for Source Classifier", type=float, default=0)
# parser.add_argument('--lr_tar_cls', help="Learning Rate for Target Classifier", type=float,
#                     default=10 ** (-4))  # Used only tuning last layer classifier weights with Target labels
# parser.add_argument('--wd_tar_cls', help="Weight Decay for Target Classifier", type=float,
#                     default=0)  # Used only tuning last layer classifier weights with Target labels
# parser.add_argument('--epochs', type=int, default=501, help="Number of Epochs")
# parser.add_argument('--target_recons_wait_epochs', type=int, default=0,
#                     help="Used only When Domain == Target, i.e., training target pipeline")
# parser.add_argument('--target_tune_classifier_epochs', type=int, default=5,
#                     help="Used only When Domain == Target and a few Target labels are provided for fine-tuning classifier's last layer weights")
# parser.add_argument('--opt_gd', type=str, default='Adam', help="Optimizer Name for Generator-Discriminator Pair")
# parser.add_argument('--opt_ed_src', type=str, default='Adam', help="Optimizer Name for Source Encoder-Decoder Pair")
# parser.add_argument('--opt_ed_tar', type=str, default='Adam', help="Optimizer Name for Target Encoder-Decoder Pair")
# parser.add_argument('--opt_cls', type=str, default='Adam', help="Optimizer Name for Classifier")
#
# # LR Scheduling Related Arguments -->
# parser.add_argument('--schdl_opt_enc_src', type=str, help="Scheduler for Source Encoder",
#                     default=None)  # list with info --> [eta_min,eta_max,ad_fac=Ascend/Descend,min_epoch=10,max_epoch=50]
# parser.add_argument('--schdl_opt_enc_tar', type=str, help="Scheduler for Target Encoder", default=None)
# parser.add_argument('--schdl_opt_dec_src', type=str, help="Scheduler for Source Decoder", default=None)
# parser.add_argument('--schdl_opt_dec_tar', type=str, help="Scheduler for Target Decoder", default=None)
# parser.add_argument('--schdl_opt_disc', type=str, help="Scheduler for Discriminator", default=None)
# parser.add_argument('--schdl_opt_gen', type=str, help="Scheduler for Generator", default=None)
# parser.add_argument('--schdl_opt_cls', type=str, help="Scheduler for Classifier", default=None)
# parser.add_argument('--schdl_opt_cls_tar', type=str, help="Scheduler for Target Classifier",
#                     default=None)  # Used only when tuning classifier weights for outer layer neuron added while classifier tuning on Target Labels
# parser.add_argument('--schdl_opt_cls_enc', type=str,
#                     help="Scheduler for Source Encoder while getting Backprop error from Classifier", default=None)
# parser.add_argument('--prnt_lr_updates', type=bool,
#                     help="To Print After Each Epoch Updated Learning Rates if using a Scheduler (Suggested: True)",
#                     default=True)
#
# # Basic Hyperparameters -->
# parser.add_argument('--bs', help="Batch Size", type=int, default=32)
# parser.add_argument('--ld', help="Latent Dimension Size", type=int, default=64)
# parser.add_argument('--tsql', type=int, help="Length used by Adaptive Pooling for making Batches of Raw Data",
#                     default=None)
# parser.add_argument('--data_type', type=str, help="Which Data to Experiment with (Raw/HCF)", default='Raw')
# parser.add_argument('--train_tar_rat', type=str, default=None, help="Ratio of Target Training Data to be used")
#
# # Loading Arguments -->
# parser.add_argument('--load_id', type=str, default=None, help="JobID corresponding to trained Source Pipeline Models")
# parser.add_argument('--load_id2', type=str, default=None, help="JobID corresponding to trained Target Pipeline Models")
# parser.add_argument('--load_mn_src_enc', type=str, default='Encoder_src',
#                     help="Name of Source Encoder's state_dict.pt to be loaded")
# parser.add_argument('--load_mn_tar_enc', type=str, default='Encoder_tar',
#                     help="Name of Target Encoder's state_dict.pt to be loaded")
# parser.add_argument('--load_mn_src_dec', type=str, default='Decoder_src',
#                     help="Name of Source Decoder's state_dict.pt to be loaded")
# parser.add_argument('--load_mn2', type=str, default='Classifier',
#                     help="Name of Classifier's state_dict.pt to be loaded")
# parser.add_argument('--load_mn2_tar', type=str, default="Classifier",
#                     help="Name of Classifier's (from target pipeline) state_dict.pt to be loaded")
# parser.add_argument('--load_pre_train_enc', type=str, default='All_layers',
#                     help="Load Source Encoder Weights as Target Encoder Weights before starting training the target pipeline")  # options=["None","All_layers","Except_first_layer"]
# parser.add_argument('--freeze_enc_tar_till', type=str, default=None,
#                     help="Number of epochs until which only Target Encoder's last layer is trained, Default: None which means train all the target encoder's weights from beginning")
# parser.add_argument('--load_pre_train_dec', type=bool, default=False,
#                     help="Whether or not to load pre-trained weights from Source Decoder (if any) into Target Decoder (if any)")
#
# # Utility Function(s) related Arguments -->
# parser.add_argument('--tsne', type=str, default='Both', help="Domains for which t-SNE plots are required")
# parser.add_argument('--threshold_selection_criterion', type=str, default='spec_sens_norm',
#                     help="How to pick the best threshold from the Eval Set Labels")
# parser.add_argument('--threshold_penalty_weight', type=float, default=0.1,
#                     help="Weight of penalty imposed for picking a threshold with specificity-sensitivity values too off from each other")
# parser.add_argument('--save_embeds', type=bool, default=False, help="To save the embeddings")
# parser.add_argument('--tsne_labels', type=int, default=1,
#                     help="Plot t-SNEs for True labels(1)/Set-point params(2)/Both(0)")  # 0 --> Both; 1 --> tru lbls only; 2 --> sp only
# parser.add_argument('--data_norm_lvl', type=str, default='Local', help="Normaliza Data on Local/Global Level")
# parser.add_argument('--data_norm_mode', type=str, default=None,
#                     help="Type of Normalization")  # options=["Standard_Scalar","Min_Max"]
# parser.add_argument('--impose_scalers', type=bool, default=False,
#                     help="To impose Normalization params calculated from Source Data on Target Data")
# parser.add_argument('--temp_thresh', type=float, default=2001,
#                     help="Used with Raw Data for not considering temperature column values above temp_thresh while calculating Normalization params")
# parser.add_argument('--upsample', type=str, default=None,
#                     help="Method to be used for upsampling HCF Data")  # To be used while loading HCF Data # options = [SMOTE,SMOTEEN]
# parser.add_argument('--prnt_processing_info', type=bool, default=False,
#                     help="To Print all the processing steps while processing HCF Data, Suggested: False for brevity")  # To be used while loading HCF Data
# parser.add_argument('--seed_upsampler', type=bool, default=False,
#                     help="To give upsampler same seed as used globally throughout the code, else it'll seed the sampler as 42")  # To be used while loading HCF Data
# parser.add_argument('--use_loader_sampler', type=bool, default=False,
#                     help="To use Imbalance Data Sampler for Raw data")  # To be used while loading Raw Data
# parser.add_argument('--want_probab_plots', type=bool, default=False, help="To plot probability plots")
#
# # Adversarial Training Related -->
# parser.add_argument('--recons_loss_func', type=str, default='MSE', help="Loss Function used for Reconstruction")
# parser.add_argument('--gan_mode', type=str, default='WGAN',
#                     help="Type of GAN required")  # options=["SNGAN","GAN","WGAN"]
# parser.add_argument('--lpchz_mode', type=str, default='GP',
#                     help="Method used for imposing the Lipchitz Condition (Applicable only for WGANs)")
# parser.add_argument('--gp_lambda', type=float, default=11,
#                     help="if lpchz_mode == GP, then gp_lambda denotes the weight of gradient penalty")
# parser.add_argument('--opt_type', type=str, default='max_min', help="Type of Optimization for WGAN")
# parser.add_argument('--f1d', type=int, default=512,
#                     help="Number of nuerons in the FIRST layer of Discriminator Architecture")
# parser.add_argument('--f2d', type=int, default=256,
#                     help="Number of nuerons in the SECOND layer of Discriminator Architecture")
# parser.add_argument('--f3d', type=int, default=128,
#                     help="Number of nuerons in the THIRD layer of Discriminator Architecture")
# parser.add_argument('--critic_iters', type=int, default=-1,
#                     help="Number of Discriminator/Critic updates per Generator update, defaults to standard values for diff types of GANs")
# parser.add_argument('--clp_l', type=float, default=-0.01,
#                     help="Lower Value for Clipping Weights (in WGAN + Weight Clipping)")
# parser.add_argument('--clp_u', type=float, default=0.01,
#                     help="Upper Value for Clipping Weights (in WGAN + Weight Clipping)")
# parser.add_argument('--enc_layers', type=int, default=3, help="Number of layers in the Encoder Architecture")
# parser.add_argument('--dec_layers', type=int, default=None,
#                     help="Number of layers in the Decoder Architecture, if None then sets it equal to the number of encoder layers")
# parser.add_argument('--enc_layers_info', type=str, default='[256,128]',
#                     help="Number of nuerons in each layer of Encoder Architecture, used with HCF Data since the Encoder Architecture's a MLP for HCF Data")  # Used while loading HCF Data
# parser.add_argument('--dec_layers_info', type=str, default=None,
#                     help="Number of nuerons in each layer of Decoder Architecture, used with HCF Data since the Decoder Architecture's a MLP for HCF Data, if set to None then it approximately mimics the encoder architecture in reverse")  # Used while loading HCF Data
# parser.add_argument('--gen_iters', type=int, default=1,
#                     help="Number of generator updates per Discriminator/Critic Update")
# parser.add_argument('--adv_pen', type=float, default=None, help="Weight given to Adversarial Penalty")
# parser.add_argument('--adv_pen_perc', type=float, default=50,
#                     help="Percentage of weight given to Adversarial Penalty, used when adv_pen isn't supplied explicitly")
# parser.add_argument('--gan_sl', type=float, default=1,
#                     help="Smooth Label to used for more stable GAN training (Applicable only with OG GANs)")
#
# # Classifier Related Arguments -->
# parser.add_argument('--classifier_loss_func', type=str, default='WFL', help="Loss Function used for Classification")
# parser.add_argument('--wfl_alpha', type=float, default=0.07, help="Alpha for WFL (if used as classifier_loss_func)")
# parser.add_argument('--wfl_gamma', type=int, default=2, help="Gamma for WFL (if used as classifier_loss_func)")
# parser.add_argument('--f1', type=str, default=None,
#                     help="Number of nuerons in first layer(s) of classifier(s) (Plural in case of Ensemble Classifier)")
# parser.add_argument('--f2', type=str, default=None,
#                     help="Number of nuerons in second layer(s) of classifier(s) (Plural in case of Ensemble Classifier)")
# parser.add_argument('--f3', type=str, default=None,
#                     help="Number of nuerons in third layer(s) of classifier(s) (Plural in case of Ensemble Classifier)")
# parser.add_argument('--def_f1', type=int, default=256,
#                     help="Default of number of neurons in first layer(s) of the classifier(s) (Applicable in case --f1 is None)")
# parser.add_argument('--def_f2', type=int, default=128,
#                     help="Default of number of neurons in second layer(s) of the classifier(s) (Applicable in case --f2 is None)")
# parser.add_argument('--def_f3', type=int, default=64,
#                     help="Default of number of neurons in third layer(s) of the classifier(s) (Applicable in case --f3 is None)")
# parser.add_argument('--ensemble', type=bool, default=False, help="True if classifier's an ensemble of MLPs")
# parser.add_argument('--model_mode', type=str, default=None,
#                     help="Ensemble classifier's mode of making predictions")  # options=["Majority_Vote","Prod_Probabs"]
# parser.add_argument('--N', type=int, default=3,
#                     help="No of 3-layered MLP classifiers to be used in the Ensemble Classifier")
# parser.add_argument('--lam', type=str, default='0.1',
#                     help="Threshold to be used on classifier outputs (A list if the classifier's an ensemble) (This threshold is to be provided if not training and just loading the classifier weights)")  # Example --> [source_thresh,target_thresh] (just_infer and if there are 2 nuerons in classifier for separate domains)
# parser.add_argument('--cls_pen', type=float, default=None,
#                     help="Weight given to Classification Penalty (if None => cls_pen set to 1)")
# parser.add_argument('--classify_every_epoch', type=bool, default=True,
#                     help="Whether to infer with Classifier on target data after every epoch or not (Used while training only target encoder)")
# parser.add_argument('--window_classifier_configs', type=str, default='[256,128]',
#                     help="If using Window Classifier as a Decoder, then pass number of neurons in each layer in the form of a list")
#
# # Misc -->
# parser.add_argument('--jobid', type=str, default="TEST_ID", help="A Unique JobID associated with each run of main.py")
# parser.add_argument('--only_deterministic', type=bool, default=False,
#                     help="True to set pytorch's engine to only_deterministic mode (For reproducibility)")
# parser.add_argument('--rand_seed', type=int, default=-1, help="If set to -1, randomly generates a seed and prints it")
# parser.add_argument('--mn1_src', type=str, default='Encoder_src', help="Name of Source Encoder's state_dict.pt")
# parser.add_argument('--mn1_tar', type=str, default='Encoder_tar', help="Name of Target Encoder's state_dict.pt")
# parser.add_argument('--mn2_src', type=str, default='Decoder_src',
#                     help="Name of Source Decoder's state_dict.pt (if any)")
# parser.add_argument('--mn2_tar', type=str, default='Decoder_tar',
#                     help="Name of Target Decoder's state_dict.pt (if any)")
# parser.add_argument('--mn3', type=str, default='Discriminator',
#                     help="Name of Discriminator's state_dict.pt (Used only while training the target pipeline)")
# parser.add_argument('--mn4', type=str, default='Classifier', help="Name of Classifier's state_dict.pt")
#
# # Domain Adaptation Related -->
# parser.add_argument('--train_mode', type=str,
#                     default='Source_pipeline')  # options=["Source_pipeline","Target_pipeline"]
# parser.add_argument('--recons', type=str, default="Win_Zeros_4",
#                     help="Type of Decoder to be used (Window Classifier or LSTM based Reconstructor)")  # if using window classification --> "Win_windowMode_windowNum" where "windowMode" can be "Zeros/Noise" and "windowNum" is an int denoting number of classes; if using LSTM-based --> "LSTM"
# parser.add_argument('--tune_tar_with_cls', type=bool, default=False,
#                     help="To use Target labels for fine-tuning Classifier weights")
#
# args = parser.parse_args()
#
# print(vars(args))

# with open(r'./configs/Wheel/tvarit_wheel.yaml', 'w') as file:
#     documents = yaml.dump(args, file)

def get_models(args, model_type='encoder'):
    Enc_Lyrs = args.enc_layers

    global_configs = get_tvarit_raw(args)

    encoder_checkpoint_path = './Models/Encoder_src.pt'

    if model_type == 'encoder':
        Encoder_src = models.Encoder_LSTM(n_features=global_configs['Source_no_of_features'], embedding_dim=global_configs['latent_dim'],N=Enc_Lyrs)
        Encoder_src.load_state_dict(torch.load(encoder_checkpoint_path))
        f1,f2,f3 = args.def_f1,args.def_f2,args.def_f3
        Classifier = models.MLP_model(f1=f1,f2=f2,f3=f3,ld=global_configs['latent_dim'])
    elif model_type == 'lstm':
        Encoder_src = None
        Classifier = models.LSTM_Classifier(n_features=global_configs['Source_no_of_features'], embedding_dim=global_configs['latent_dim'],N=Enc_Lyrs)
    elif model_type == 'resnet':
        Encoder_src = None
        Classifier = models.ResNet(n_in=301, n_classes=2)
    elif model_type == 'tsc':
        Encoder_src = None
        Classifier = models.TimeSeriesCnnModel()

    return global_configs, Encoder_src, Classifier
