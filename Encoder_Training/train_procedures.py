# %%
# *********************** Importing Essential Libraries *************************
import torch
import time
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

import train_utilities as train_utilities
import test_procedures as test_procedures
import utilis as utilis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# ************* Random Seed to all the devices ****************
def rand_seed_devices(configs):    
    torch.backends.cudnn.deterministic = configs['only_deterministic']
    torch.manual_seed(configs['rand_seed'])
    np.random.seed(configs['rand_seed'])
    random.seed(configs['rand_seed'])

# **************************************************** End-to-end training functions ********************************************************

# %%
def train_source_pipeline(Encoder_src,Decoder_src,Classifier,configs):
    
    logger_dict={} # To Store all the Losses & Accuracies 
    logger_dict['Train'] = {}
    logger_dict['Test'] = {}
    logger_dict['Source'] = {} # Stores everything related to Classifier Pipeline

    if configs['recons_domain']=="Source":
        logger_dict['Train']['Recons_Loss_src'] = []
        logger_dict['Test']['Recons_Loss_src'] = []
        if configs['recons_window']:
            logger_dict['Train']['Accuracy_src'] = []
            logger_dict['Test']['Accuracy_src'] = []
    else:
        logger_dict['Train']['Recons_Loss_src'] = [-1]
        logger_dict['Test']['Recons_Loss_src'] = [-1]
    
    logger_dict['Source']['Train_Loss'] = []
    logger_dict['Source']['Eval_Loss'] = []
    logger_dict['Source']['Test_Loss'] = []
    logger_dict['Source']['Train_Sensitivity'] = []
    logger_dict['Source']['Eval_Sensitivity'] = []
    logger_dict['Source']['Test_Sensitivity'] = []
    logger_dict['Source']['Train_Specificity'] = []
    logger_dict['Source']['Eval_Specificity'] = []
    logger_dict['Source']['Test_Specificity'] = []
    logger_dict['Source']['Train_Precision'] = []
    logger_dict['Source']['Eval_Precision'] = []
    logger_dict['Source']['Test_Precision'] = []
    logger_dict['Source']['Train_F1_Score'] = []
    logger_dict['Source']['Eval_F1_Score'] = []
    logger_dict['Source']['Test_F1_Score'] = []
    logger_dict['Source']['Train_AUC'] = []
    logger_dict['Source']['Eval_AUC'] = []
    logger_dict['Source']['Test_AUC'] = []

    logger_dict['Best_Thresholds'] = []

    for epoch in range(configs['epochs']):
        configs['curr_epoch']=epoch
        utilis.opt_steps(configs=configs) # For Learning Rate Scheduling 
        start_time = time.time()
        losses_recons = 0 # Reconstruction Loss per epoch
        losses_classifier=0 # Classifier Loss per epoch

        for i,(src_tuple) in enumerate(configs['Source_Train_Loader']): 
            whl_src = src_tuple[0]
            src_y = src_tuple[1].to(device)

            if configs['recons_window']:
                whl_src_pert,src_y_pert = src_tuple[-2],src_tuple[-1].to(device) # Perturbed Wheel data with its (perturbed) label
                source_x_pert = Variable(whl_src_pert.to(device),requires_grad=False)
            source_x = Variable(whl_src.to(device),requires_grad=False)
            src_y = src_y.view(-1)
            #src_y_pert = src_y_pert.view(-1)

            # Training the Source Encoder -->    
            if configs['recons_domain'] == 'Source':
                if configs['recons_window']:
                    Decoder_src,Encoder_src,loss = train_utilities.train_classifier_encoder_batch(Model=Decoder_src,Encoder=Encoder_src,X=source_x_pert,y=src_y_pert,configs=configs,mode='Window_src')
                else:
                    Encoder_src,Decoder_src,loss = train_utilities.train_encoder_decoder_batch(Encoder=Encoder_src,Decoder=Decoder_src,X=source_x,configs=configs,is_tar=False)
                losses_recons+=loss
            
            # Training the Classifier --> 
            Classifier,Encoder_src,loss_cls = train_utilities.train_classifier_encoder_batch(Model=Classifier,Encoder=Encoder_src,X=source_x,y=src_y,configs=configs)
            losses_classifier+=loss_cls          

        # Collecting Scores after each epoch -->
        if configs['recons_domain'] == 'Source': 
            logger_dict['Train']['Recons_Loss_src'].append(losses_recons.cpu().detach().numpy()/(i+1))
        logger_dict['Source']['Train_Loss'].append(losses_classifier.cpu().detach().numpy()/(i+1))
        print('Epoch [{}/{}], Time Taken: {:.2f}s, Encoder_Window_Classifier Train Loss (Source): {:.5f}, Encoder_Window_Classifier Classifier Loss (Source): {:.5f}'.format(epoch + 1, configs['epochs'],time.time()-start_time,logger_dict['Train']['Recons_Loss_src'][-1],logger_dict['Source']['Train_Loss'][-1]))
    """
        # Computing Metrics on Train, Eval, Test Sets after switching on model's eval mode -->
        temp_configs = configs.copy()  # Shallow Copy of configs
        temp_configs['Source_Train_Loader']=None
        src_train_loader = test_procedures.encoder_forward_pass(Encoder=Encoder_src,loader=configs['Source_Train_Loader'],blen=configs['batch_size'])
        temp_configs['Source_Train_Loader'] = src_train_loader
        if configs['recons_domain'] == 'Source':
            logger_dict,src_eval_loader,src_test_loader = test_procedures.test_ae_src_batch(Models=[Encoder_src,Decoder_src],configs=configs,logger=logger_dict,ret_loaders=True)
        else:
            src_eval_loader = test_procedures.encoder_forward_pass(Encoder=Encoder_src,loader=configs['Source_Eval_Loader'],blen=configs['batch_size'])
            src_test_loader = test_procedures.encoder_forward_pass(Encoder=Encoder_src,loader=configs['Source_Test_Loader'],blen=configs['batch_size'])
        temp_configs['Source_Eval_Loader'] = src_eval_loader
        temp_configs['Source_Test_Loader'] = src_test_loader
        
        temp_configs['Target_Train_Loader'] = None
        temp_configs['Target_Eval_Loader'] = None
        temp_configs['Target_Test_Loader'] = None
        logger_dict = test_procedures.test_classifier_batch(logger=logger_dict,Model=Classifier,configs=temp_configs)
        if configs['recons_window']:
            logger_dict = test_procedures.test_window_classifier_batch(logger=logger_dict,Models=[Encoder_src,Decoder_src],configs=configs)
                
        # Printing All the Scores per epoch -->
        if configs['recons_window']:
            print('Epoch [{}/{}], Time Taken: {:.2f}s, Encoder_Window_Classifier Train Loss (Source): {:.5f}, Encoder_Window_Classifier Train Accuracy (Source): {:.2f}%, Encoder_Window_Classifier Test Loss (Source): {:.5f}, Encoder_Window_Classifier Test Accuracy (Source): {:.2f}%'.format(epoch + 1, configs['epochs'],time.time()-start_time,logger_dict['Train']['Recons_Loss_src'][-1],logger_dict['Train']['Accuracy_src'][-1],logger_dict['Test']['Recons_Loss_src'][-1],logger_dict['Test']['Accuracy_src'][-1]))
        else:
            print('Epoch [{}/{}], Time Taken: {:.2f}s, Encoder_Decoder Train Loss (Source): {:.5f}, Encoder_Decoder Test Loss (Source): {:.5f}'.format(epoch + 1, configs['epochs'],time.time()-start_time,logger_dict['Train']['Recons_Loss_src'][-1],logger_dict['Test']['Recons_Loss_src'][-1]))
        print("")
        print("Classification Results on Source Data --> ")
        print('Train Loss: {:.5f}, Train Specificity: {:.2f}%, Train Sensitivity: {:.2f}%, Train Precision: {:.2f}%, Train F1_Score: {:.2f}%, Train AUC_Score: {:.2f}%, Eval Loss: {:.5f}, Eval Specificity: {:.2f}%, Eval Sensitivity: {:.2f}%, Eval Precision: {:.2f}%, Eval F1_Score: {:.2f}%, Eval AUC_Score: {:.2f}%, Test Loss: {:.5f}, Test Specificity: {:.2f}%, Test Sensitivity: {:.2f}%, Test Precision: {:.2f}%, Test F1_Score: {:.2f}%, Test AUC_Score: {:.2f}%'.format(logger_dict['Source']['Train_Loss'][-1],logger_dict['Source']['Train_Specificity'][-1],logger_dict['Source']['Train_Sensitivity'][-1],logger_dict['Source']['Train_Precision'][-1],logger_dict['Source']['Train_F1_Score'][-1],logger_dict['Source']['Train_AUC'][-1],logger_dict['Source']['Eval_Loss'][-1],logger_dict['Source']['Eval_Specificity'][-1],logger_dict['Source']['Eval_Sensitivity'][-1],logger_dict['Source']['Eval_Precision'][-1],logger_dict['Source']['Eval_F1_Score'][-1],logger_dict['Source']['Eval_AUC'][-1],logger_dict['Source']['Test_Loss'][-1],logger_dict['Source']['Test_Specificity'][-1],logger_dict['Source']['Test_Sensitivity'][-1],logger_dict['Source']['Test_Precision'][-1],logger_dict['Source']['Test_F1_Score'][-1],logger_dict['Source']['Test_AUC'][-1]))
        print("")         
        print("Best Thresh: "+str(logger_dict['Best_Thresholds'][-1]))
        print("")
    """

    # Saving the model weights --->
    torch.save(Encoder_src.state_dict(),configs['save_encoder_src_path'])
    if Decoder_src is not None:
        torch.save(Decoder_src.state_dict(),configs['save_decoder_src_path'])
    torch.save(Classifier.state_dict(),configs['save_classifier_path'])
    """
    # Plotting Classification Related Learning Curve(s) -->
    utilis.plot_train_val_curve(logger_dict['Source']['Train_Loss'],[logger_dict['Source']['Eval_Loss'],logger_dict['Source']['Test_Loss']],configs['epochs'],nm='Classification_Loss',is_percent=False,path=configs['source_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Source']['Train_Specificity'],[logger_dict['Source']['Eval_Specificity'],logger_dict['Source']['Test_Specificity']],configs['epochs'],path=configs['source_plot_path'],nm='Specificity')
    utilis.plot_train_val_curve(logger_dict['Source']['Train_Sensitivity'],[logger_dict['Source']['Eval_Sensitivity'],logger_dict['Source']['Test_Sensitivity']],configs['epochs'],nm='Sensitivity',path=configs['source_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Source']['Train_Precision'],[logger_dict['Source']['Eval_Precision'],logger_dict['Source']['Test_Precision']],configs['epochs'],nm='Precision',path=configs['source_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Source']['Train_F1_Score'],[logger_dict['Source']['Eval_F1_Score'],logger_dict['Source']['Test_F1_Score']],configs['epochs'],nm='F1_Score',path=configs['source_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Source']['Train_AUC'],[logger_dict['Source']['Eval_AUC'],logger_dict['Source']['Test_AUC']],configs['epochs'],nm='AUC_Score',path=configs['source_plot_path'])

    if configs['is_ensemble']:
        if configs['ensemble_mode']!='Majority_Vote':
            utilis.plot_train_val_curve(logger_dict['Best_Thresholds'],[],configs['epochs'],nm='Best_Thresholds',is_percent=False,path=configs['save_plot_path'])
        else:
            best_lams_list = np.asarray(logger_dict['Best_Thresholds'])
            for i in range(configs['ensemble_models']):
                utilis.plot_train_val_curve(best_lams_list[:,i],[],configs['epochs'],nm='Best_Thresholds_Model'+str(i+1),is_percent=False,path=configs['save_plot_path'])
    else:
        utilis.plot_train_val_curve(logger_dict['Best_Thresholds'],[],configs['epochs'],nm='Best_Thresholds',is_percent=False,path=configs['save_plot_path'])

    # Plotting the Reconstruction Losses Curve(s) (if any) -->
    
    if configs['recons_window']:
        utilis.plot_train_val_curve(logger_dict['Train']['Recons_Loss_src'],logger_dict['Test']['Recons_Loss_src'],configs['epochs'],nm='Source_Encoder_Window_Classifier_Loss',is_percent=False,path=configs['save_plot_path'])
        utilis.plot_train_val_curve(logger_dict['Train']['Accuracy_src'],logger_dict['Test']['Accuracy_src'],configs['epochs'],nm='Source_Encoder_Window_Classifier_Accuracy',is_percent=True,path=configs['save_plot_path'])    
    else:
        if configs['recons_domain'] == 'Source':
            utilis.plot_train_val_curve(logger_dict['Train']['Recons_Loss_src'],logger_dict['Test']['Recons_Loss_src'],configs['epochs'],nm='Source_Encoder_Decoder_Loss',is_percent=False,path=configs['save_plot_path'])
    """
    return Encoder_src,Decoder_src,Classifier#,logger_dict['Best_Thresholds'][-1]

# %%
def train_target_pipeline(Encoder_src,Encoder_tar,Decoder_tar,Classifier,Discriminator,configs): # Decoder_tar = None implies only Adversarial Training Required
    logger_dict={}
    logger_dict['Train'] = {}
    logger_dict['Test'] = {}

    logger_dict['Train']['Gen_Loss'] = []
    logger_dict['Test']['Gen_Loss'] = []
    logger_dict['Train']['Disc_Loss'] = []
    logger_dict['Test']['Disc_Loss'] = []
    logger_dict['Train']['Disc_Tru_Loss'] = []
    logger_dict['Test']['Disc_Tru_Loss'] = []
    logger_dict['Train']['Disc_Fake_Loss'] = []
    logger_dict['Test']['Disc_Fake_Loss'] = []
    logger_dict['Train']['Recons_Loss_tar'] = [-1] # Would remain -1 signifying that Decoder_tar is None
    logger_dict['Test']['Recons_Loss_tar'] = [-1] # Would remain -1 signifying that Decoder_tar is None
    if configs['recons_window']: # Decoder_tar is a window classifier
        logger_dict['Train']['Accuracy_tar'] = [-1]
        logger_dict['Test']['Accuracy_tar'] = [-1]
        
    if configs['wgan_train_mode']=='GP' and configs['gan_mode']=='WGAN': # GP = Gradient Penalty
        logger_dict['Train']['GP_Loss'] = []

    if configs['classify_every_epoch']: # Whether to calculate and print classification scores on target sets after each epoch or not; Default = True (Useful for Early Stopping)
        if "best_threshold" in configs.keys():
            logger_dict['Best_Thresholds'] = [configs['best_threshold']]
        logger_dict['Target'] = {}
        logger_dict['Target']['Train_Loss'] = []
        logger_dict['Target']['Eval_Loss'] = []
        logger_dict['Target']['Test_Loss'] = []
        logger_dict['Target']['Train_Sensitivity'] = []
        logger_dict['Target']['Eval_Sensitivity'] = []
        logger_dict['Target']['Test_Sensitivity'] = []
        logger_dict['Target']['Train_Specificity'] = []
        logger_dict['Target']['Eval_Specificity'] = []
        logger_dict['Target']['Test_Specificity'] = []
        logger_dict['Target']['Train_Precision'] = []
        logger_dict['Target']['Eval_Precision'] = []
        logger_dict['Target']['Test_Precision'] = []
        logger_dict['Target']['Train_F1_Score'] = []
        logger_dict['Target']['Eval_F1_Score'] = []
        logger_dict['Target']['Test_F1_Score'] = []
        logger_dict['Target']['Train_AUC'] = []
        logger_dict['Target']['Eval_AUC'] = []
        logger_dict['Target']['Test_AUC'] = []
        if configs['train_cls_tar']: # Whether or not to train classifier's last nueron on TARGET LABELS; if TARGET LABELS to be used 
            logger_dict['Target']['Best_Thresholds'] = []

    for epoch in range(configs['epochs']):
        configs['curr_epoch']=epoch
        utilis.opt_steps(configs=configs) # For Learning Rate Scheduling 
        configs,Encoder_tar = utilis.unfreeze_lyrs(Model=Encoder_tar,configs=configs,model_nm='Encoder_tar')
        start_time = time.time()
        d_tru_err=d_false_err=0
        if configs['wgan_train_mode']=='GP' and configs['gan_mode']=='WGAN':
            grad_pen_err=0
        g_losses = 0 # Generator Losses per Epoch
        d_losses = 0 # Critic/Discriminator Losses per Epoch
        if Decoder_tar is not None:
            recons_losses=0 # Reconstruction/Decoder's Loss per Epoch

        for i,(src_tuple,tar_tuple) in enumerate(zip(configs['Source_Train_Loader'],configs['Target_Train_Loader'])): 
            whl_src,whl_tar = src_tuple[0],tar_tuple[0]

            if configs['recons_window']:
                whl_tar_pert,tar_y_pert = tar_tuple[-2],tar_tuple[-1].to(device)

            if whl_src.shape[0]!=whl_tar.shape[0]:   # Making sure that both the batches are of same sizes
                sml_len = min(whl_src.shape[0],whl_tar.shape[0])
                if whl_src.shape[0]<whl_tar.shape[0]:
                    whl_tar = whl_tar[:(sml_len)]
                    if configs['recons_window']:
                        whl_tar_pert = whl_tar_pert[:(sml_len)]
                        tar_y_pert = tar_y_pert[:(sml_len)]
                else:
                    whl_src = whl_src[:(sml_len)]

            source_x = Variable(whl_src.to(device),requires_grad=False)
            target_x = Variable(whl_tar.to(device),requires_grad=False)
            if configs['recons_window']:
                target_x_pert = Variable(whl_tar_pert.to(device),requires_grad=False)

            # Training DISCRIMINATOR --> 
            if configs['gan_mode']=='WGAN':
                if configs['wgan_train_mode']=='GP':
                    Discriminator,err_D_fake,err_D_tru,err_D_penalty,loss_d = train_utilities.train_wgan_disc_batch(Discriminator=Discriminator,Generator=[Encoder_src,Encoder_tar],source_x=source_x,target_x=target_x,configs=configs,double_gen=True)
                    grad_pen_err+=err_D_penalty
                else:
                    Discriminator,err_D_fake,err_D_tru,loss_d = train_utilities.train_wgan_disc_batch(Discriminator=Discriminator,Generator=[Encoder_src,Encoder_tar],source_x=source_x,target_x=target_x,configs=configs,double_gen=True)
            elif configs['gan_mode']=='GAN':
                Discriminator,err_D_fake,err_D_tru,loss_d = train_utilities.train_gan_disc_batch(Discriminator=Discriminator,Generator=[Encoder_src,Encoder_tar],source_x=source_x,target_x=target_x,configs=configs,double_gen=True)
            
            d_losses+=loss_d
            d_tru_err+=err_D_tru
            d_false_err+=err_D_fake

            # Training GENERATOR --> 
            Encoder_tar,loss_g = train_utilities.train_generator_batch(Discriminator=Discriminator,Generator=Encoder_tar,target_x=target_x,configs=configs)
            g_losses+=loss_g  

            # Training DECODER_TAR (if any) --> 
            if (Decoder_tar is not None) and epoch>=configs['tar_epochs_wait']: # tar_epochs_wait --> Initial epochs to wait before training Decoder_tar (if any); Default = 0
                if configs['recons_window']:
                    Decoder_tar,Encoder_tar,loss = train_utilities.train_classifier_encoder_batch(Model=Decoder_tar,Encoder=Encoder_tar,X=target_x_pert,y=tar_y_pert,configs=configs,mode='Window_tar')
                else:
                    Encoder_tar,Decoder_tar,loss = train_utilities.train_encoder_decoder_batch(Encoder=Encoder_tar,Decoder=Decoder_tar,X=target_x,configs=configs,is_tar=True)
                recons_losses+=loss      

        if configs['train_cls_tar']: # Train Classifier's last nueron on TARGET LABELS if required
            inner_train_loader = test_procedures.encoder_forward_pass(Encoder=Encoder_tar,loader=configs['Target_Train_lbls_Loader'],blen=configs['batch_size'])
            for inner_epoch in range(configs['cls_tar_epochs']): # cls_tar_epochs => No of Epochs (typically b/w 1 to 5) for which Classifier's last nueron is to be trained!
                for i, (inner_data_tuple) in enumerate(inner_train_loader): # batches --> (blen,latent_dim)
                    # Run the forward pass
                    batches,labels = inner_data_tuple[0],inner_data_tuple[1]
                    batches = Variable(batches.to(device),requires_grad=False)
                    labels = labels.to(device)
                    Classifier,loss_cls = train_utilities.train_classifier_batch(Model=Classifier,X=batches,y=labels,configs=configs,is_tar=True)

        # Collecting Scores after each epoch -->
        logger_dict['Train']['Gen_Loss'].append(g_losses.cpu().detach().numpy()/(i+1))
        logger_dict['Train']['Disc_Loss'].append(d_losses.cpu().detach().numpy()/(i+1))
        logger_dict['Train']['Disc_Tru_Loss'].append(d_tru_err.cpu().detach().numpy()/(i+1))
        logger_dict['Train']['Disc_Fake_Loss'].append(d_false_err.cpu().detach().numpy()/(i+1))
        if configs['wgan_train_mode']=='GP' and configs['gan_mode']=='WGAN':
            logger_dict['Train']['GP_Loss'].append(grad_pen_err.cpu().detach().numpy()/(i+1))
        if Decoder_tar is not None and epoch>=configs['tar_epochs_wait']:
            logger_dict['Train']['Recons_Loss_tar'].append(recons_losses.cpu().detach().numpy()/(i+1))

        # Computing Classification Scores after switching on Eval mode -->
        if configs['classify_every_epoch']:
            temp_configs = configs.copy()  # Shallow Copy of configs
            # We don't want to compute scores on Source Datasets
            temp_configs['Source_Train_Loader']=None
            temp_configs['Source_Eval_Loader']=None
            temp_configs['Source_Test_Loader']=None

            tar_train_loader = test_procedures.encoder_forward_pass(Encoder=Encoder_tar,loader=configs['Target_Train_Loader'],blen=configs['batch_size'])
            temp_configs['Target_Train_Loader'] = tar_train_loader
            recons_mode='None'
            if Decoder_tar is not None and epoch>=configs['tar_epochs_wait']:
                recons_mode='Target'
            logger_dict,_,tar_eval_loader,___,tar_test_loader = test_procedures.test_adv_recons_batch(Models=[Encoder_src,Encoder_tar,Discriminator,Decoder_tar],configs=configs,logger=logger_dict,recons_mode=recons_mode,ret_loaders=True)
            temp_configs['Target_Eval_Loader'] = tar_eval_loader
            temp_configs['Target_Test_Loader'] = tar_test_loader

            if configs['train_cls_tar']:
                temp_configs['Target_Test_lbls_Loader'] = test_procedures.encoder_forward_pass(Encoder=Encoder_tar,loader=configs['Target_Test_lbls_Loader'],blen=configs['batch_size'])
                logger_dict = test_procedures.test_classifier_tar_batch(logger=logger_dict,Model=Classifier,configs=temp_configs)
            else:
                logger_dict = test_procedures.test_classifier_batch(logger=logger_dict,Model=Classifier,configs=temp_configs)
        else:
            recons_mode='None'
            if Decoder_tar is not None and epoch>=configs['tar_epochs_wait']:
                recons_mode='Target'
            logger_dict,_,__,___,____ = test_procedures.test_adv_recons_batch(Models=[Encoder_src,Encoder_tar,Discriminator,Decoder_tar],configs=configs,logger=logger_dict,recons_mode=recons_mode)
            
        if configs['recons_window'] and epoch>=configs['tar_epochs_wait']:
            logger_dict = test_procedures.test_window_classifier_batch(logger=logger_dict,Models=[Encoder_tar,Decoder_tar],configs=configs)

        # Printing All the Scores per epoch -->
        if not configs['recons_window']:
            recons_prnt_wrd_sr = "Encoder_Decoder"
            key_for_losses_sr="Recons"

        if configs['recons_window']:
            if configs['wgan_train_mode']!='GP' or configs['gan_mode']=='GAN':
                print('Epoch [{}/{}], Time Taken: {:.2f}s, Encoder_Window_Classifier Train Loss (Target): {:.5f}, Encoder_Window_Classifier Train Accuracy (Target): {:.2f}%, Generator Train Loss: {:.5f}, Discriminator Train Loss: {:.5f}, Discriminator Train Loss (True): {:.3f}, Discriminator Train Loss (Fake): {:.3f}, Encoder_Window_Classifier Test Loss (Target): {:.5f}, Encoder_Window_Classifier Test Accuracy (Target): {:.2f}%, Generator Test Loss: {:.5f}, Discriminator Test Loss: {:.5f}, Discriminator Test Loss (True): {:.3f}, Discriminator Test Loss (Fake): {:.3f}'.format(epoch + 1, configs['epochs'],time.time()-start_time,logger_dict['Train']['Recons_Loss_tar'][-1],logger_dict['Train']['Accuracy_tar'][-1],logger_dict['Train']['Gen_Loss'][-1],logger_dict['Train']['Disc_Loss'][-1],logger_dict['Train']['Disc_Tru_Loss'][-1],logger_dict['Train']['Disc_Fake_Loss'][-1],logger_dict['Test']['Recons_Loss_tar'][-1],logger_dict['Test']['Accuracy_tar'][-1],logger_dict['Test']['Gen_Loss'][-1],logger_dict['Test']['Disc_Loss'][-1],logger_dict['Test']['Disc_Tru_Loss'][-1],logger_dict['Test']['Disc_Fake_Loss'][-1]))
            else:
                print('Epoch [{}/{}], Time Taken: {:.2f}s, Encoder_Window_Classifier Train Loss (Target): {:.5f}, Encoder_Window_Classifier Train Accuracy (Target): {:.2f}%, Generator Train Loss: {:.5f}, Discriminator Train Loss: {:.5f}, Discriminator Train Loss (True): {:.3f}, Discriminator Train Loss (Fake): {:.3f}, Discriminator Train Gradient Penalty Loss: {:.5f}, Encoder_Window_Classifier Test Loss (Target): {:.5f}, Encoder_Window_Classifier Test Accuracy (Target): {:.2f}%, Generator Test Loss: {:.5f}, Discriminator Test Loss: {:.5f}, Discriminator Test Loss (True): {:.3f}, Discriminator Test Loss (Fake): {:.3f}'.format(epoch + 1, configs['epochs'],time.time()-start_time,logger_dict['Train']['Recons_Loss_tar'][-1],logger_dict['Train']['Accuracy_tar'][-1],logger_dict['Train']['Gen_Loss'][-1],logger_dict['Train']['Disc_Loss'][-1],logger_dict['Train']['Disc_Tru_Loss'][-1],logger_dict['Train']['Disc_Fake_Loss'][-1],logger_dict['Train']['GP_Loss'][-1],logger_dict['Test']['Recons_Loss_tar'][-1],logger_dict['Test']['Accuracy_tar'][-1],logger_dict['Test']['Gen_Loss'][-1],logger_dict['Test']['Disc_Loss'][-1],logger_dict['Test']['Disc_Tru_Loss'][-1],logger_dict['Test']['Disc_Fake_Loss'][-1]))
        else:
            if configs['wgan_train_mode']!='GP' or configs['gan_mode']=='GAN':
                print('Epoch [{}/{}], Time Taken: {:.2f}s, {} Train Loss (Target): {:.5f}, Generator Train Loss: {:.5f}, Discriminator Train Loss: {:.5f}, Discriminator Train Loss (True): {:.3f}, Discriminator Train Loss (Fake): {:.3f}, {} Test Loss (Target): {:.5f}, Generator Test Loss: {:.5f}, Discriminator Test Loss: {:.5f}, Discriminator Test Loss (True): {:.3f}, Discriminator Test Loss (Fake): {:.3f}'.format(epoch + 1, configs['epochs'],time.time()-start_time,recons_prnt_wrd_sr,logger_dict['Train'][key_for_losses_sr+'_Loss_tar'][-1],logger_dict['Train']['Gen_Loss'][-1],logger_dict['Train']['Disc_Loss'][-1],logger_dict['Train']['Disc_Tru_Loss'][-1],logger_dict['Train']['Disc_Fake_Loss'][-1],recons_prnt_wrd_sr,logger_dict['Test'][key_for_losses_sr+'_Loss_tar'][-1],logger_dict['Test']['Gen_Loss'][-1],logger_dict['Test']['Disc_Loss'][-1],logger_dict['Test']['Disc_Tru_Loss'][-1],logger_dict['Test']['Disc_Fake_Loss'][-1]))
            else:
                print('Epoch [{}/{}], Time Taken: {:.2f}s, {} Train Loss (Target): {:.5f}, Generator Train Loss: {:.5f}, Discriminator Train Loss: {:.5f}, Discriminator Train Loss (True): {:.3f}, Discriminator Train Loss (Fake): {:.3f}, Discriminator Train Gradient Penalty Loss: {:.5f}, {} Test Loss (Target): {:.5f}, Generator Test Loss: {:.5f}, Discriminator Test Loss: {:.5f}, Discriminator Test Loss (True): {:.3f}, Discriminator Test Loss (Fake): {:.3f}'.format(epoch + 1, configs['epochs'],time.time()-start_time,recons_prnt_wrd_sr,logger_dict['Train'][key_for_losses_sr+'_Loss_tar'][-1],logger_dict['Train']['Gen_Loss'][-1],logger_dict['Train']['Disc_Loss'][-1],logger_dict['Train']['Disc_Tru_Loss'][-1],logger_dict['Train']['Disc_Fake_Loss'][-1],logger_dict['Train']['GP_Loss'][-1],recons_prnt_wrd_sr,logger_dict['Test'][key_for_losses_sr+'_Loss_tar'][-1],logger_dict['Test']['Gen_Loss'][-1],logger_dict['Test']['Disc_Loss'][-1],logger_dict['Test']['Disc_Tru_Loss'][-1],logger_dict['Test']['Disc_Fake_Loss'][-1]))    
        print("")
        if configs['classify_every_epoch']:
            print("Results on Target Data --> ")
            print('Train Loss: {:.5f}, Train Specificity: {:.2f}%, Train Sensitivity: {:.2f}%, Train Precision: {:.2f}%, Train F1_Score: {:.2f}%, Train AUC_Score: {:.2f}%, Eval Loss: {:.5f}, Eval Specificity: {:.2f}%, Eval Sensitivity: {:.2f}%, Eval Precision: {:.2f}%, Eval F1_Score: {:.2f}%, Eval AUC_Score: {:.2f}%, Test Loss: {:.5f}, Test Specificity: {:.2f}%, Test Sensitivity: {:.2f}%, Test Precision: {:.2f}%, Test F1_Score: {:.2f}%, Test AUC_Score: {:.2f}%'.format(logger_dict['Target']['Train_Loss'][-1],logger_dict['Target']['Train_Specificity'][-1],logger_dict['Target']['Train_Sensitivity'][-1],logger_dict['Target']['Train_Precision'][-1],logger_dict['Target']['Train_F1_Score'][-1],logger_dict['Target']['Train_AUC'][-1],logger_dict['Target']['Eval_Loss'][-1],logger_dict['Target']['Eval_Specificity'][-1],logger_dict['Target']['Eval_Sensitivity'][-1],logger_dict['Target']['Eval_Precision'][-1],logger_dict['Target']['Eval_F1_Score'][-1],logger_dict['Target']['Eval_AUC'][-1],logger_dict['Target']['Test_Loss'][-1],logger_dict['Target']['Test_Specificity'][-1],logger_dict['Target']['Test_Sensitivity'][-1],logger_dict['Target']['Test_Precision'][-1],logger_dict['Target']['Test_F1_Score'][-1],logger_dict['Target']['Test_AUC'][-1]))
            if configs['train_cls_tar']:
                print("")
                print("Best Thresh (Train_Target): "+str(logger_dict['Target']['Best_Thresholds'][-1]))
            print("")

    # Saving the model weights --->
    torch.save(Encoder_tar.state_dict(),configs['save_encoder_tar_path'])
    torch.save(Discriminator.state_dict(),configs['save_discriminator_path'])
    if Decoder_tar is not None:
        torch.save(Decoder_tar.state_dict(),configs['save_decoder_tar_path'])
    if configs['train_cls_tar']:
        torch.save(Classifier.state_dict(),configs['save_classifier_path'])

    # Plotting the Training Loss Curve(s) -->
    utilis.plot_train_val_curve(logger_dict['Train']['Gen_Loss'],logger_dict['Test']['Gen_Loss'],configs['epochs'],nm='Generator_Loss',is_percent=False,path=configs['save_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Train']['Disc_Loss'],logger_dict['Test']['Disc_Loss'],configs['epochs'],nm='Discriminator_Loss',is_percent=False,path=configs['save_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Train']['Disc_Tru_Loss'],logger_dict['Test']['Disc_Tru_Loss'],configs['epochs'],nm='Discriminator_Loss_True',is_percent=False,path=configs['save_plot_path'])
    utilis.plot_train_val_curve(logger_dict['Train']['Disc_Fake_Loss'],logger_dict['Test']['Disc_Fake_Loss'],configs['epochs'],nm='Discriminator_Loss_Fake',is_percent=False,path=configs['save_plot_path'])
    if configs['wgan_train_mode']=='GP' and configs['gan_mode']=='WGAN':
        utilis.plot_train_val_curve(logger_dict['Train']['GP_Loss'],[],configs['epochs'],nm='Discriminator_Penalty_Loss',is_percent=False,path=configs['save_plot_path'])
    if configs['recons_window']:
        utilis.plot_train_val_curve(logger_dict['Train']['Recons_Loss_tar'][1:],logger_dict['Test']['Recons_Loss_tar'][1:],configs['epochs']-configs['tar_epochs_wait'],nm='Target_Encoder_Window_Classifier_Loss',is_percent=False,path=configs['save_plot_path'])
        utilis.plot_train_val_curve(logger_dict['Train']['Accuracy_tar'][1:],logger_dict['Test']['Accuracy_tar'][1:],configs['epochs']-configs['tar_epochs_wait'],nm='Target_Encoder_Window_Classifier_Accuracy',is_percent=True,path=configs['save_plot_path'])
    else:
        if configs['recons_domain'] == 'Target':
            utilis.plot_train_val_curve(logger_dict['Train'][key_for_losses_sr+'_Loss_tar'][1:],logger_dict['Test'][key_for_losses_sr+'_Loss_tar'][1:],configs['epochs']-configs['tar_epochs_wait'],nm='Target_'+recons_prnt_wrd_sr+'_Loss',is_percent=False,path=configs['save_plot_path'])
    
    if configs['classify_every_epoch']:
        tar_plot_epochs = configs['epochs']
        utilis.plot_train_val_curve(logger_dict['Target']['Train_Loss'],[logger_dict['Target']['Eval_Loss'],logger_dict['Target']['Test_Loss']],tar_plot_epochs,nm='Classification_Loss',is_percent=False,path=configs['target_plot_path'])
        utilis.plot_train_val_curve(logger_dict['Target']['Train_Specificity'],[logger_dict['Target']['Eval_Specificity'],logger_dict['Target']['Test_Specificity']],tar_plot_epochs,path=configs['target_plot_path'],nm='Specificity')
        utilis.plot_train_val_curve(logger_dict['Target']['Train_Sensitivity'],[logger_dict['Target']['Eval_Sensitivity'],logger_dict['Target']['Test_Sensitivity']],tar_plot_epochs,nm='Sensitivity',path=configs['target_plot_path'])
        utilis.plot_train_val_curve(logger_dict['Target']['Train_Precision'],[logger_dict['Target']['Eval_Precision'],logger_dict['Target']['Test_Precision']],tar_plot_epochs,nm='Precision',path=configs['target_plot_path'])
        utilis.plot_train_val_curve(logger_dict['Target']['Train_F1_Score'],[logger_dict['Target']['Eval_F1_Score'],logger_dict['Target']['Test_F1_Score']],tar_plot_epochs,nm='F1_Score',path=configs['target_plot_path'])
        utilis.plot_train_val_curve(logger_dict['Target']['Train_AUC'],[logger_dict['Target']['Eval_AUC'],logger_dict['Target']['Test_AUC']],tar_plot_epochs,nm='AUC_Score',path=configs['target_plot_path'])
        
        if configs['train_cls_tar']:
            if configs['is_ensemble']:
                if configs['ensemble_mode']!='Majority_Vote':
                    utilis.plot_train_val_curve(logger_dict['Target']['Best_Thresholds'],[],tar_plot_epochs,nm='Target_Best_Thresholds',is_percent=False,path=configs['save_plot_path'])
                else:
                    best_lams_list = np.asarray(logger_dict['Target']['Best_Thresholds'])
                    for i in range(configs['ensemble_models']):
                        utilis.plot_train_val_curve(best_lams_list[:,i],[],tar_plot_epochs,nm='Target_Best_Thresholds_Model'+str(i+1),is_percent=False,path=configs['save_plot_path'])
            else:
                utilis.plot_train_val_curve(logger_dict['Target']['Best_Thresholds'],[],tar_plot_epochs,nm='Target_Best_Thresholds',is_percent=False,path=configs['save_plot_path'])

    if configs['train_cls_tar']:
        return Encoder_tar,Decoder_tar,Discriminator,Classifier,logger_dict['Target']['Best_Thresholds'][-1]
    else:
        return Encoder_tar,Decoder_tar,Discriminator,Classifier,None
