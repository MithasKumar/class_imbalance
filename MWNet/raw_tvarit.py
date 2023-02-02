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
import utilis
import resnet
import data_processing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device name :- " + str(device))


def get_tvarit_raw(args):
    # %%
    # *********************** Standard ML Hyper-parameters *************************
    # learning_rate1 = args.lr_src_enc_dec
    # learning_rate1_tar = args.lr_tar_enc_dec
    # learning_rate1_G = args.lr_tar_gen
    # learning_rate1_D = args.lr_disc
    # learning_rate1_cls = args.lr_src_enc_cls
    # weight_dr1 = args.wd_src_enc_dec
    # learning_rate2 = args.lr_src_cls
    # weight_dr2 = args.wd_src_cls
    # learning_rate2_tar = args.lr_tar_cls
    # weight_dr2_tar = args.wd_tar_cls
    # lr_configs = {}
    # lr_configs['optimizer_classifier'] = learning_rate2
    # lr_configs['optimizer_classifier_tar'] = learning_rate2_tar
    # lr_configs['optimizer_encoder_src'] = learning_rate1
    # lr_configs['optimizer_decoder_src'] = learning_rate1
    # lr_configs['optimizer_encoder_tar'] = learning_rate1_tar
    # lr_configs['optimizer_decoder_tar'] = learning_rate1_tar
    # lr_configs['optimizer_generator'] = learning_rate1_G
    # lr_configs['optimizer_critic'] = learning_rate1_D
    # lr_configs['optimizer_encoder_classifier'] = learning_rate1_cls

    # t-SNE related variables -->
    to_plt_tsne = args.tsne
    if to_plt_tsne == "Both" and args.train_mode == "Source_pipeline":
        to_plt_tsne = "Source"
    want_sp = want_lbls = True
    if args.tsne_labels == 1:
        want_sp = False
    if args.tsne_labels == 2:
        want_lbls = False

    # just_infer = (args.load_id is not None) and (args.load_id2 is not None)  # True if main.py is ran just for inferring
    # if just_infer:
    #     args.train_mode = None
    #     recons_domain = None

    # # Encoder and Decoder Architecture Related Arguments -->
    Enc_Lyrs = args.enc_layers
    # if args.dec_layers is None:
    Dec_Lyrs = Enc_Lyrs
    # else:
    #     Dec_Lyrs = args.dec_layers

    # # Decoder related Variables -->
    recons_window = args.recons.split("_")[0] == "Win"
#    if not just_infer:
    recons_domain = args.train_mode.split("_")[0]

    # **************** Some Relevant Paths **************************
    root_path = "./"  # CHANGE THIS TO Appropriate ROOT PATH
    data_path = os.path.join(root_path, "Datasets")
    model_path = os.path.join(root_path, "Models")
    save_model_path = os.path.join(model_path, args.jobid)
    save_plot_path = os.path.join(root_path, "Plots", args.jobid)
    exist_ok_val = True
    os.makedirs(save_plot_path, exist_ok=exist_ok_val)
    os.makedirs(save_model_path, exist_ok=exist_ok_val)
    
    # %%
    # ************** Initializing Global Configs Dict ********************
    global_configs = {}
    #SPECUFY CONFIGS AND DATASET NAMES HERE!!
    global_configs['Source_configs_path'] = os.path.join(data_path, 'cza_modelling_data_config.json')
    global_configs['Target_configs_path'] = os.path.join(data_path, 'da_target_configs.json')
    global_configs['Source_train_data_path'] = os.path.join(data_path, 'cza_classifier_train.feather')
    global_configs['Source_eval_data_path'] = os.path.join(data_path, 'cza_classifier_val.feather')
    global_configs['Source_test_data_path'] = os.path.join(data_path, 'cza_classifier_test.feather')
    global_configs['Target_train_data_path'] = os.path.join(data_path, 'da_target_train.feather')
    global_configs['Target_eval_data_path'] = os.path.join(data_path, 'da_target_eval.feather')
    global_configs['Target_test_data_path'] = os.path.join(data_path, 'da_target_test.feather')
    # if args.data_type == 'HCF':
    #     global_configs['data_path'] = data_path
    #     # global_configs['domain'] = args.train_mode.split("_")[0]
    #     global_configs['upsample'] = args.upsample
    #     global_configs['seed_upsample'] = args.seed_upsampler
    #     global_configs['prnt_processing_info'] = args.prnt_processing_info
    #     global_configs['Source_no_of_features'] = 599
    global_configs['batch_size'] = args.bs
    global_configs['latent_dim'] = args.ld
    global_configs['jobid'] = args.jobid
    global_configs['enc_layers'] = args.enc_layers
    if args.data_type == 'Raw':
        global_configs['use_loader_sampler'] = args.use_loader_sampler
    global_configs['to_save_embeds'] = args.save_embeds
    if global_configs['to_save_embeds']:
        save_embeds_path = os.path.join(save_plot_path, 'Embeddings')
        os.makedirs(save_embeds_path, exist_ok=exist_ok_val)
        global_configs['save_train_embeds_path'] = os.path.join(save_embeds_path, 'both_domains_train_embeds.csv')
        global_configs['save_test_embeds_path'] = os.path.join(save_embeds_path, 'both_domains_test_embeds.csv')
    # if args.tune_tar_with_cls or recons_domain == "Source":
    #     global_configs['save_classifier_path'] = os.path.join(save_model_path, args.mn4 + '.pt')
    #     global_configs['source_plot_path'] = os.path.join(save_plot_path, 'Source_Plots')
    #     global_configs['target_plot_path'] = os.path.join(save_plot_path, 'Target_Plots')
    #     os.makedirs(global_configs['source_plot_path'], exist_ok=exist_ok_val)
    #     os.makedirs(global_configs['target_plot_path'], exist_ok=exist_ok_val)

    # if not just_infer:
    global_configs['epochs'] = args.epochs
        # if args.train_mode.split("_")[0] == "Source":
        #     global_configs['save_encoder_src_path'] = os.path.join(save_model_path, args.mn1_src + '.pt')
        #     global_configs['save_decoder_src_path'] = os.path.join(save_model_path, args.mn2_src + '.pt')
        # elif args.train_mode.split("_")[0] == "Target":
        #     global_configs['save_encoder_tar_path'] = os.path.join(save_model_path, args.mn1_tar + '.pt')
        #     global_configs['save_decoder_tar_path'] = os.path.join(save_model_path, args.mn2_tar + '.pt')
        #     global_configs['save_discriminator_path'] = os.path.join(save_model_path, args.mn3 + '.pt')

    # if args.critic_iters != -1:
    #     global_configs['critic_iters'] = args.critic_iters
    # else:
    #     if args.gan_mode in ['GAN', 'SNGAN']:
    #         global_configs['critic_iters'] = 1
    #     elif args.gan_mode == 'WGAN':
    #         global_configs['critic_iters'] = 5
    global_configs['want_sp'] = want_sp
    global_configs['want_lbls'] = want_lbls
    # global_configs['unfreeze_first_lyr_only'] = False

    # if recons_domain == "Target":
    #     global_configs['tar_epochs_wait'] = args.target_recons_wait_epochs
    #     if args.tune_tar_with_cls:
    #         global_configs['cls_tar_epochs'] = args.target_tune_classifier_epochs
    #     if args.freeze_enc_tar_till is not None:
    #         global_configs['unfreeze_first_lyr_only'] = True
    #         global_configs['unfreeze_epochs'] = int(args.freeze_enc_tar_till)

    # global_configs['wgan_train_mode'] = args.lpchz_mode
    # global_configs['gp_penalty'] = args.gp_lambda
    if args.data_type == 'Raw':
        global_configs['adap_len'] = args.tsql
        if args.tsql is None:
            global_configs['adap_len'] = 301
    global_configs['min_max'] = args.opt_type == 'min_max'
    # if args.adv_pen is None:
    #     global_configs['adv_pen'] = args.adv_pen_perc / 100
    #     global_configs['cls_pen'] = 1 - global_configs['adv_pen']
    # else:
    #     global_configs['adv_pen'] = args.adv_pen
    #     if args.cls_pen is not None:
    #         global_configs['cls_pen'] = args.cls_pen
    #     else:
    #         global_configs['cls_pen'] = 1
    # global_configs['is_ensemble'] = args.ensemble
    # global_configs['ensemble_models'] = args.N
    # global_configs['ensemble_mode'] = args.model_mode
    # global_configs['recons_loss_func'] = args.recons_loss_func
    # global_configs['clip_upper'] = args.clp_u
    # global_configs['clip_lower'] = args.clp_l

    # global_configs['classifier_loss_func'] = args.classifier_loss_func
    # global_configs['wfl_alpha'] = args.wfl_alpha
    # global_configs['wfl_gamma'] = args.wfl_gamma
    # global_configs['thresh_selection'] = args.threshold_selection_criterion
    # global_configs['thresh_pen_weight'] = args.threshold_penalty_weight
    # if global_configs['thresh_selection'] == 'spec_sens':
    #     global_configs['thresh_pen_weight'] = 0
    # print("")
    # print("Penalty used for optimal threshold selection = " + str(global_configs['thresh_pen_weight']))
    # print("")

    # if args.classify_every_epoch and recons_domain == "Target":
    #     global_configs['target_plot_path'] = os.path.join(save_plot_path, 'Target_Plots')
    #     os.makedirs(global_configs['target_plot_path'], exist_ok=exist_ok_val)
    global_configs['save_plot_path'] = save_plot_path
    global_configs['to_plt_tsne'] = to_plt_tsne
    if to_plt_tsne != 'None' and global_configs['want_sp']:
        global_configs['save_src_tsne_plots_path'] = os.path.join(save_plot_path, 't-SNE_Plots', 'Source')
        global_configs['save_tar_tsne_plots_path'] = os.path.join(save_plot_path, 't-SNE_Plots', 'Target')
        os.makedirs(global_configs['save_src_tsne_plots_path'], exist_ok=exist_ok_val)
        os.makedirs(global_configs['save_tar_tsne_plots_path'], exist_ok=exist_ok_val)
    global_configs['only_deterministic'] = args.only_deterministic
    if args.rand_seed != -1:
        global_configs['rand_seed'] = args.rand_seed
    else:
        global_configs['rand_seed'] = random.randint(0, (2 ** 32) - 1)
        print("Using Random Seed ==> " + str(global_configs['rand_seed']))

    # global_configs['gan_mode'] = args.gan_mode
    # if args.gan_mode == 'SNGAN':
    #     global_configs['gan_mode'] = 'GAN'
    # if global_configs['gan_mode'] == 'WGAN':
    #     one = torch.FloatTensor([1]).to(device)
    #     mone = one * -1
    #     global_configs['one'] = one
    #     global_configs['mone'] = mone
    # elif global_configs['gan_mode'] == 'GAN':
    #     global_configs['gan_smooth_label'] = args.gan_sl
    # global_configs['gen_iters'] = args.gen_iters

    global_configs['classify_every_epoch'] = args.classify_every_epoch
    # global_configs['train_cls_tar'] = args.tune_tar_with_cls
    # if global_configs['train_cls_tar']:
    #     global_configs['classify_every_epoch'] = True
    #     global_configs['Target_train_lbls_data_path'] = os.path.join(data_path, 'da_target_train_lbls.feather')
    #     global_configs['Target_eval_lbls_data_path'] = os.path.join(data_path, 'da_target_eval_lbls.feather')

    if args.data_type == 'Raw':
        global_configs['data_norm_lvl'] = args.data_norm_lvl
    global_configs['data_norm_mode'] = args.data_norm_mode
    if global_configs['data_norm_mode'] is None:
        if args.data_type == 'Raw':
            global_configs['data_norm_mode'] = 'Min_Max'
        elif args.data_type == 'HCF':
            global_configs['data_norm_mode'] = 'Standard_Scalar'
    global_configs['impose_scalers'] = args.impose_scalers
    # if args.data_type == 'Raw':
    #     global_configs['temp_thresh'] = args.temp_thresh
    global_configs['prnt_lr_updates'] = args.prnt_lr_updates
    global_configs['recons_window'] = recons_window
    if args.recons != "None":
        global_configs['recons_domain'] = recons_domain

    # global_configs['want_probab_plots'] = args.want_probab_plots
    # if global_configs['want_probab_plots']:
    #     save_probab_plts_path = os.path.join(save_plot_path, 'Probab_Plots')
    #     os.makedirs(save_probab_plts_path, exist_ok=exist_ok_val)
    #     global_configs['save_probab_plots_path'] = save_probab_plts_path
    if recons_window:
        global_configs["win_recons_info"] = args.recons
        global_configs['win_cls_configs'] = args.window_classifier_configs
        global_configs['cls_lyrs'] = Dec_Lyrs
    if args.data_type == 'HCF':
        dec_sig = global_configs['data_norm_mode'] == 'Min_Max'
    elif args.data_type == 'Raw':
        dec_sig = global_configs['data_norm_mode'] == 'Min_Max' and global_configs['data_norm_lvl'] == 'Local'

    if args.data_type == "Raw":
        global_configs["subset_dict"] = utilis.make_subset_dict(string=args.train_tar_rat)
    opt_lst = []
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
    # train_procedures.rand_seed_devices(configs=global_configs)
    # train_utilities.rand_seed_devices(configs=global_configs)
    # test_procedures.rand_seed_devices(configs=global_configs)
    utilis.rand_seed_devices(configs=global_configs)
    # inference.rand_seed_devices(configs=global_configs)
    data_processing.rand_seed_devices(configs=global_configs)
    resnet.rand_seed_devices(configs=global_configs)
    # print(sitaram())

    # %%
    if args.data_type == 'Raw':
        global_configs = data_processing.get_raw_data_df(configs=global_configs)  # Adding Raw Data's Dataframes
        # data_processing.print_raw_data_df_info(configs=global_configs)  # Print Dataset Info
        global_configs = data_processing.get_raw_data_to_loaders(configs=global_configs)
    # elif args.data_type == 'HCF':
    #     global_configs = data_processing.get_hcf_data_to_loaders(configs=global_configs)
    else:
        raise NotImplementedError("Data Type can't be recognized :(")

    # for i, (src_tuple) in enumerate(global_configs['Source_Train_Loader']):
    #     whl_src = src_tuple[0]
    #     src_y = src_tuple[1].to(device)
    #
    #     print('length', len(src_tuple))
    #
    #     if global_configs['recons_window']:
    #         whl_src_pert, src_y_pert = src_tuple[-2], src_tuple[-1].to(
    #             device)  # Perturbed Wheel data with its (perturbed) label
    #         source_x_pert = Variable(whl_src_pert.to(device), requires_grad=False)
    #     source_x = Variable(whl_src.to(device), requires_grad=False)
    return global_configs






