# %%
# *********************** Importing Essential Libraries *************************
import numpy as np
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import dataset.train_utils.utilis as utilis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# ************* Random Seed to all the devices ****************
def rand_seed_devices(configs):    
    torch.backends.cudnn.deterministic = configs['only_deterministic']
    torch.manual_seed(configs['rand_seed'])
    np.random.seed(configs['rand_seed'])
    random.seed(configs['rand_seed'])

# %%
def best_threshold(y_true, y_prob,configs,thresh=None): # For computing the best threshold 
    r''' thresh: If not None => Take this thresh also into account while picking the best threshold
    '''
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if thresh is not None:
        thresholds = np.append(thresholds,thresh)
        temp_fpr,temp_tpr = utilis.calc_tpr_fpr(y_tru=y_true,thresh=thresh,y_prob=y_prob)
        fpr = np.append(fpr,temp_fpr)
        tpr = np.append(tpr,temp_tpr)   
    if 'spec_sens' in configs['thresh_selection']:
        # calculate the g-mean of sensitivity & specificity for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        gmeans -= (configs['thresh_pen_weight']*np.abs(tpr+fpr-1))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
    elif configs['thresh_selection']=='f1_score':
        p = y_true.sum().item()
        tot = y_true.size(0)
        n = tot-p
        ratio = n/p
        f1s = (2*tpr)/(tpr+1+(ratio*fpr))
        ix = np.argmax(f1s)
    return thresholds[ix]

# %%
def metrics_calc(y,outs,lmbd=None,opt_configs=None): # For computing the classification scores; Returns in the order => Specificity, Sensitivity, Precision, F1-Score
    if lmbd is not None:
        predicted_sr = utilis.make_preds(lmbd=lmbd,probs=outs)
    else:
        predicted_sr=outs
    cfm_sr = confusion_matrix(y, predicted_sr)
    tn, fp, fn, tp = cfm_sr.ravel()
    return (tn/(tn+fp))*100,(tp/(tp+fn))*100,(tp/(tp+fp))*100,((2*tp)/(tp+fp+fn+tp))*100

# %%
def test_classifier_loader(Model,loader,configs,fwd_target,out_preds=False,keepdim=False):
    r"""
    Returns the avg classification loss, prediction_probabs and ground_truths
    keepdim: bool => To keep the last dim (avoiding ground_truths and prediction_probabs from getting squeezed to 1-d vecs)
    out_preds: bool => Used with Multi-Class Output Classifier(s) ONLY!; If True, then applies Softmax + Argmax to prediction logits and returns prediction labels
    fwd_target: bool => If True, treats Model as a Classifier with separate neuron for Target Neuron in the Last Layer 
    """
    net_classification_loss=0
    ctr=0
    Model.eval()
    flag=True
    criterion = configs['criterion_classify']
    with torch.no_grad():
        for data_tuple in loader:
            # Run the forward pass
            batches = data_tuple[0].to(device)
            labels = data_tuple[1].to(device)
            if configs['is_ensemble']:
                labels = labels.repeat_interleave(configs['ensemble_models'],1)
            if fwd_target:
                outs = Model(batches,out_mode='Target')
            else:
                outs = Model(batches)

            if out_preds:
                loss = criterion(outs, labels)
            else:
                loss = criterion(outs, labels.float())
            net_classification_loss+=loss.item()

            if out_preds:
                outs = nn.Softmax(dim=-1)(outs)
                outs = torch.argmax(outs,dim=-1,keepdim=keepdim)

            if flag:
                flag=False
                pred_probabs = outs
                ground_truths = labels
            else:
                pred_probabs = torch.cat((pred_probabs,outs),dim=0)
                ground_truths = torch.cat((ground_truths,labels),dim=0)
            
            ctr+=1
    if configs['is_ensemble']:
        if configs['ensemble_mode']=='Majority_Vote':
            return net_classification_loss/ctr,pred_probabs,ground_truths
        elif configs['ensemble_mode']=='Prod_Probabs':
            return net_classification_loss/ctr,torch.prod(pred_probabs,dim=-1,keepdim=True),torch.unsqueeze(ground_truths[:,0],dim=-1)
    else:
        return net_classification_loss/ctr,pred_probabs,ground_truths

# %%
def test_classification_scores_plots(Model,loader,configs,is_eval=False,threshold=None,plt_roc_dict=None,only_thresh=False,is_tar=False,probab_plts_dict=None):
    r"""
    Given a Model (classifier) & a loader; computes classification scores aned best threshold on that loader + plot probability plot(s) and roc curve(s) if needed 
    plt_roc_dict: A dict if not None => Will plot ROC Curve for this loader; plt_roc_dict['path'] -- path were we want to save this ROC Plot; plt_roc_dict['nm'] -- name of saved plot
    probab_plts_dict: A dict if not None => Will plot Probaility Plots for this loader; keys mean the same as "plt_roc_dict" 
    is_eval: bool => If True, would consider the loader as an eval_loader and compute best_threshold on it
    threshold: float/list of floats => The best threhold(s) used for evaluation (used when is_eval=False)
    only_thresh: Just compute the best threshold(s) and return w/o computing the scores; works only when is_eval=True
    is_tar: bool => Whether the loader belongs to Target domain or not; used when we have a separate neuron in the last layer for Target Domain (trained with TARGET LABELS) 
    """
    # best_lams => stores the best threshold computed; if computed

    if loader is None:
        return None,None,None,None,None,None,None

    plt_roc=False
    if plt_roc_dict is not None:
        plt_roc=True
        path = plt_roc_dict['path']
        nm = plt_roc_dict['nm']

    want_probab_plts=False
    if probab_plts_dict is not None:
        want_probab_plts=True
        pp_path = probab_plts_dict['path']
        pp_nm = probab_plts_dict['nm']

    l,prediction_probabilities,ground_truths = test_classifier_loader(Model=Model,loader=loader,configs=configs,fwd_target=is_tar) # l = loss
    ground_truths,prediction_probabilities = ground_truths.cpu(),prediction_probabilities.cpu()

    if want_probab_plts and not configs['is_ensemble']:
        utilis.plotProbs(gt=torch.squeeze(ground_truths,dim=-1).numpy(), pred=torch.squeeze(prediction_probabilities,dim=-1).numpy(), path=pp_path, nm=pp_nm)

    if configs['is_ensemble']:

        if is_eval or threshold is None:   # Choose the best threshold from the eval set 
            if configs['ensemble_mode']=='Majority_Vote':
                best_lams = []
                for i in range(configs['ensemble_models']):
                    gt,probs = ground_truths[:,i],prediction_probabilities[:,i]
                    best_lams.append(best_threshold(y_true=gt, y_prob=probs,configs=configs))
            else:
                gt,probs = ground_truths[:,0],prediction_probabilities[:,0]
                best_lams = best_threshold(y_true=gt,y_prob=probs,configs=configs)
            if only_thresh:
                return best_lams

        if configs['ensemble_mode']!='Majority_Vote':
            auc = roc_auc_score(ground_truths.view(-1),prediction_probabilities.view(-1))*100
            spec,sens,prec,f1 = metrics_calc(y=ground_truths.view(-1),outs=prediction_probabilities.view(-1),lmbd=best_lams)
            if plt_roc:
                utilis.plot_roc_curve(y_true=ground_truths, y_prob=prediction_probabilities,nm=nm,path=path,best_thresh=best_lams)
        else:
            avg_auc = 0
            for i in range(configs['ensemble_models']):
                gt,probs = ground_truths[:,i],prediction_probabilities[:,i]
                if plt_roc:
                    utilis.plot_roc_curve(y_true=gt, y_prob=probs,nm=str(nm)+'_'+str(i+1),path=path,best_thresh=best_lams[i])
                avg_auc += roc_auc_score(gt,probs)*100
                preds_temp = torch.where(probs>=best_lams[i],torch.Tensor([1]),torch.Tensor([0]))
                if i==0:
                    real_preds = torch.unsqueeze(preds_temp,dim=-1)
                else:
                    real_preds = torch.cat((real_preds,torch.unsqueeze(preds_temp,dim=-1)),dim=1)
            real_preds = real_preds.sum(dim=-1)
            prediction_probabilities = torch.where(real_preds>=configs['ensemble_models']//2,torch.Tensor([1]),torch.Tensor([0]))
            auc = avg_auc/configs['ensemble_models']
            spec,sens,prec,f1 = metrics_calc(y=ground_truths[:,0],outs=prediction_probabilities)

    else:
        if is_eval or threshold is None:
            best_lams = best_threshold(y_true=ground_truths, y_prob=prediction_probabilities,configs=configs)
            if only_thresh:
                return best_lams
            spec,sens,prec,f1 = metrics_calc(y=ground_truths,outs=prediction_probabilities,lmbd=best_lams)
        else:
            spec,sens,prec,f1 = metrics_calc(y=ground_truths,outs=prediction_probabilities,lmbd=threshold)
        if plt_roc:
            utilis.plot_roc_curve(y_true=ground_truths, y_prob=prediction_probabilities,nm=nm,path=path,best_thresh=threshold)
        auc = roc_auc_score(ground_truths,prediction_probabilities)*100

    if is_eval or threshold is None:
        return best_lams,l,spec,sens,prec,f1,auc
    else:
        return threshold,l,spec,sens,prec,f1,auc

# %%
def test_classifier_batch(logger,Model,configs,update_train_loss=False):

    if configs['Source_Eval_Loader'] is None:
        best_lams=logger['Best_Thresholds'][-1] # Else if eval loader exists then pick the best threshold by looking at the eval loader's labels

    # Evaluating Scores on Source Set(s) -->
    if configs['Source_Eval_Loader'] is not None:
        best_lams,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Source_Eval_Loader'],configs=configs,is_eval=True)
        logger['Best_Thresholds'].append(best_lams)
        logger['Source']['Eval_Loss'].append(l)
        logger['Source']['Eval_Specificity'].append(spec)
        logger['Source']['Eval_Sensitivity'].append(sens)
        logger['Source']['Eval_Precision'].append(prec)
        logger['Source']['Eval_F1_Score'].append(f1)
        logger['Source']['Eval_AUC'].append(auc)

    if configs['Source_Train_Loader'] is not None:
        __,_,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Source_Train_Loader'],configs=configs,is_eval=False,threshold=best_lams)
        if update_train_loss:
            logger['Source']['Train_Loss'].append(_)
        logger['Source']['Train_Specificity'].append(spec)
        logger['Source']['Train_Sensitivity'].append(sens)
        logger['Source']['Train_Precision'].append(prec)
        logger['Source']['Train_F1_Score'].append(f1)
        logger['Source']['Train_AUC'].append(auc)

    if configs['Source_Test_Loader'] is not None:
        _,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Source_Test_Loader'],configs=configs,is_eval=False,threshold=best_lams)
        logger['Source']['Test_Loss'].append(l)
        logger['Source']['Test_Specificity'].append(spec)
        logger['Source']['Test_Sensitivity'].append(sens)
        logger['Source']['Test_Precision'].append(prec)
        logger['Source']['Test_F1_Score'].append(f1)
        logger['Source']['Test_AUC'].append(auc)

    # Evaluating Scores on Target Set(s) -->
    best_lams_tar = best_lams
    first_test_loader = 'Train'
    second_test_loader = 'Eval'
    # if configs['tune_tar']: # Pick the appropriate threshold first (by looking at available Target Labels in Target Domain's Eval Loader)
    #     first_test_loader = 'Eval'
    #     second_test_loader = 'Train'
    #     best_lams_tar=None

    if configs['Target_'+first_test_loader+'_Loader'] is not None:
        thresh_new,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Target_'+first_test_loader+'_Loader'],configs=configs,is_eval=False,threshold=best_lams_tar)
        # if configs['tune_tar']:
        #     logger['Target']['Best_Thresholds'].append(thresh_new)
        #     best_lams_tar = thresh_new
        logger['Target'][first_test_loader+'_Loss'].append(l)
        logger['Target'][first_test_loader+'_Specificity'].append(spec)
        logger['Target'][first_test_loader+'_Sensitivity'].append(sens)
        logger['Target'][first_test_loader+'_Precision'].append(prec)
        logger['Target'][first_test_loader+'_F1_Score'].append(f1)
        logger['Target'][first_test_loader+'_AUC'].append(auc)

    if configs['Target_'+second_test_loader+'_Loader'] is not None:
        _,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Target_'+second_test_loader+'_Loader'],configs=configs,is_eval=False,threshold=best_lams_tar)
        logger['Target'][second_test_loader+'_Loss'].append(l)
        logger['Target'][second_test_loader+'_Specificity'].append(spec)
        logger['Target'][second_test_loader+'_Sensitivity'].append(sens)
        logger['Target'][second_test_loader+'_Precision'].append(prec)
        logger['Target'][second_test_loader+'_F1_Score'].append(f1)
        logger['Target'][second_test_loader+'_AUC'].append(auc)

    if configs['Target_Test_Loader'] is not None:
        _,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Target_Test_Loader'],configs=configs,is_eval=False,threshold=best_lams_tar)
        logger['Target']['Test_Loss'].append(l)
        logger['Target']['Test_Specificity'].append(spec)
        logger['Target']['Test_Sensitivity'].append(sens)
        logger['Target']['Test_Precision'].append(prec)
        logger['Target']['Test_F1_Score'].append(f1)
        logger['Target']['Test_AUC'].append(auc)

    return logger

# %%
def test_classifier_tar_batch(logger,Model,configs,update_train_loss=True):

    # Picking Best Thresh on Target Eval -->
    best_lams = test_classification_scores_plots(Model=Model,loader=configs['Target_Test_lbls_Loader'],configs=configs,is_eval=True,only_thresh=True,is_tar=True)
    logger['Target']['Best_Thresholds'].append(best_lams)

    # Evaluating on Target Sets -->
    _,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Target_Train_Loader'],configs=configs,is_eval=False,threshold=best_lams,is_tar=True)
    if update_train_loss:
        logger['Target']['Train_Loss'].append(l)
    logger['Target']['Train_Specificity'].append(spec)
    logger['Target']['Train_Sensitivity'].append(sens)
    logger['Target']['Train_Precision'].append(prec)
    logger['Target']['Train_F1_Score'].append(f1)
    logger['Target']['Train_AUC'].append(auc)

    _,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Target_Eval_Loader'],configs=configs,is_eval=False,threshold=best_lams,is_tar=True)
    logger['Target']['Eval_Loss'].append(l)
    logger['Target']['Eval_Specificity'].append(spec)
    logger['Target']['Eval_Sensitivity'].append(sens)
    logger['Target']['Eval_Precision'].append(prec)
    logger['Target']['Eval_F1_Score'].append(f1)
    logger['Target']['Eval_AUC'].append(auc)

    _,l,spec,sens,prec,f1,auc = test_classification_scores_plots(Model=Model,loader=configs['Target_Test_Loader'],configs=configs,is_eval=False,threshold=best_lams,is_tar=True)
    logger['Target']['Test_Loss'].append(l)
    logger['Target']['Test_Specificity'].append(spec)
    logger['Target']['Test_Sensitivity'].append(sens)
    logger['Target']['Test_Precision'].append(prec)
    logger['Target']['Test_F1_Score'].append(f1)
    logger['Target']['Test_AUC'].append(auc)

    return logger

# %%
def test_window_classifier_loader(Models,loader,configs,compute_loss=False,out_preds=True):
    net_classification_loss=0
    flag=True
    if compute_loss:
        criterion=configs['criterion_recons']
    E,D = Models[0],Models[-1]
    E.eval()
    D.eval()
    for i,(data_tuple) in enumerate(loader):
        X,y = data_tuple[-2].to(device),data_tuple[-1].to(device)
        outs = D(E(X))
        if compute_loss:
            loss = criterion(outs,torch.squeeze(y,dim=-1))
            net_classification_loss+=loss.item()
        if out_preds:
            outs = nn.Softmax(dim=-1)(outs)
            outs = torch.argmax(outs,dim=-1,keepdim=True)
        if flag:
            flag=False
            pred_probabs = outs
            ground_truths = y
        else:
            pred_probabs = torch.cat((pred_probabs,outs),dim=0)
            ground_truths = torch.cat((ground_truths,y),dim=0)

    return net_classification_loss/(i+1),pred_probabs,ground_truths

# %%
def test_window_classifier_batch(Models,configs,logger):

    if configs['recons_domain'] == 'Source':
        _,preds,gt = test_window_classifier_loader(Models=Models,loader=configs['Source_Train_Loader'],configs=configs)
        logger['Train']['Accuracy_src'].append(utilis.calc_accuracy_frm_multi_class(y_preds=preds,y_tru=gt))
        # if configs['Source_Eval_Loader'] is not None:
        _,preds,gt = test_window_classifier_loader(Models=Models,loader=configs['Source_Eval_Loader'],configs=configs)
        _,preds2,gt2 = test_window_classifier_loader(Models=Models,loader=configs['Source_Test_Loader'],configs=configs)
        preds = torch.cat((preds,preds2),dim=0)
        gt = torch.cat((gt,gt2),dim=0)
        # else:
        #     _,preds,gt = test_window_classifier_loader(Models=Models,loader=configs['Source_Test_Loader'],configs=configs)
        logger['Test']['Accuracy_src'].append(utilis.calc_accuracy_frm_multi_class(y_preds=preds,y_tru=gt))

    elif configs['recons_domain'] == 'Target':
        _,preds,gt = test_window_classifier_loader(Models=Models,loader=configs['Target_Train_Loader'],configs=configs)
        logger['Train']['Accuracy_tar'].append(utilis.calc_accuracy_frm_multi_class(y_preds=preds,y_tru=gt))
        # if configs['Target_Eval_Loader'] is not None:
        _,preds,gt = test_window_classifier_loader(Models=Models,loader=configs['Target_Eval_Loader'],configs=configs)
        _,preds2,gt2 = test_window_classifier_loader(Models=Models,loader=configs['Target_Test_Loader'],configs=configs)
        preds = torch.cat((preds,preds2),dim=0)
        gt = torch.cat((gt,gt2),dim=0)
        # else:
            # _,preds,gt = test_window_classifier_loader(Models=Models,loader=configs['Target_Test_Loader'],configs=configs)
        logger['Test']['Accuracy_tar'].append(utilis.calc_accuracy_frm_multi_class(y_preds=preds,y_tru=gt))

    else:
        raise NotImplementedError("Unable to decipher recons_domain in function test_window_classifier_batch() :(") 

    return logger

# %%
def encoder_forward_pass(Encoder,loader,blen=32,to_shfl=False,return_dataset=False):
    r"""
    Given loader => returns data_loader which is just loader projected to latent space by the Encoder
    return_dataset: bool -> to just return projected (latent) data 
    """
    if loader is None:
        return loader
    Encoder.eval()
    flag=True
    with torch.no_grad():
        for i,(data_tuple) in enumerate(loader):
            x = data_tuple[0].to(device)
            label = data_tuple[1]
            z = Encoder(x)
            if flag:
                flag=False
                batches = z
                labels = label
            else:
                batches = torch.cat((batches,z),dim=0)
                labels = torch.cat((labels,label),dim=0)
    if return_dataset:
        return batches
    data_set = TensorDataset(batches,labels)
    data_loader = DataLoader(dataset=data_set, batch_size=blen, shuffle=to_shfl)
    return data_loader

# %%
def test_adv_recons_loader(Models,configs,loader,frm=None,ret_loader=False,to_shfl=False,recons_mode=None):
    r"""
    Compute Adv + Recons (if applicable) Losses for a loader
    """
    Encoder,Discriminator = Models[0],Models[1]
    # if recons_mode is None:
    #     recons_mode=configs['recons_domain']
    if (frm=='Tar' and recons_mode == 'Target'):
    # if recons_mode == "Target":
        Decoder = Models[-1]
        Decoder.eval()
    Encoder.eval()
    Discriminator.eval()

    if ret_loader:
        flag=True

    if configs['gan_mode']=='WGAN':
        mult_fac=-1
        if frm=='Src':
            mult_fac=1
    elif configs['gan_mode']=='GAN':
        smooth_label = configs['gan_smooth_label']
        criterion = nn.BCELoss().to(device)
        d_losses=0

    losses_ed=0
    d_tru_err=d_false_err=0
    g_losses=0

    with torch.no_grad():
        
        for i,(data_tuple) in enumerate(loader):
            x = data_tuple[0].to(device)

            if configs['gan_mode']=='GAN':
                y_true = smooth_label*torch.ones((x.size(0), 1)).to(device)
                if frm=='Tar':
                    y_fake = torch.zeros((x.size(0), 1)).to(device)

            if ret_loader:
                label = data_tuple[1]

            z = Encoder(x)    
            # PHASE1: Reconstruction Loss -->
            if (frm=='Tar' and recons_mode == 'Target'):
                criterion = configs['criterion_recons']
                if configs['recons_window']:
                    x_ = data_tuple[-2].to(device)
                    z_ = Encoder(x_)
                    out = Decoder(z_)
                    y_ = data_tuple[-1].to(device)
                    y_ = torch.squeeze(y_,dim=-1)
                    loss = criterion(out, y_)
                else:
                    out = Decoder(z)
                    loss = criterion(out, x)
                losses_ed += (loss.item())

            # PHASE2: Regularization (Adversarial Loss) -->
            out = Discriminator(z)

            if configs['gan_mode']=='WGAN':
                err_D = mult_fac*configs['adv_pen']*torch.mean(out)
                if configs['min_max']:
                    err_D*=-1

                if frm=='Src':
                    d_tru_err += (err_D.item())
                else:
                    d_false_err += (err_D.item())
                    loss_g = -err_D
                    g_losses += (loss_g.item())

            elif configs['gan_mode']=='GAN':
                if frm=='Src':
                    d_tru_err += torch.mean(out).item()
                    d_losses += criterion(out,y_true)
                elif frm=='Tar':
                    d_false_err += torch.mean(out).item()
                    d_losses += criterion(out,y_fake)
                    g_losses += criterion(out,y_true)

            if ret_loader:
                if flag:
                    flag=False
                    batches = z
                    labels = label
                else:
                    batches = torch.cat((batches,z),dim=0)
                    labels = torch.cat((labels,label),dim=0)

    if not (ret_loader):
        if configs['gan_mode']=='WGAN':
            return losses_ed,g_losses,d_tru_err,d_false_err,i,None
        elif configs['gan_mode']=='GAN':
            return losses_ed,g_losses,d_losses,d_tru_err,d_false_err,i,None
    else:
        data_set = TensorDataset(batches,labels)
        data_loader = DataLoader(dataset=data_set, batch_size=configs['batch_size'], shuffle=to_shfl)
        if configs['gan_mode']=='WGAN':
            return losses_ed,g_losses,d_tru_err,d_false_err,i,data_loader
        elif configs['gan_mode']=='GAN':
            return losses_ed,g_losses,d_losses,d_tru_err,d_false_err,i,data_loader

# %%
def test_adv_recons_batch(Models,configs,logger,recons_mode,ret_loaders=False): # Models --> A list of models in the following the sequence --> [Encoder_src,Encoder_tar,Discriminator,Decoder_tar]
    r"""
    ret_loaders: bool => Return Data Loaders for Source+Target Eval & Test Sets; with Data Loaders themselves in latent space

    """
    src_eval_loader=src_test_loader=None
    losses_ed_src = losses_ed_tar = 0
    d_tru_err=d_false_err=0
    g_losses = 0
    if configs['gan_mode']=='GAN':
        d_losses=0 # Discriminator Losses
    Models_src = [Models[0],Models[2]]
    Models_tar = [Models[1],Models[2],Models[3]]
    j=0

    if configs['Source_Eval_Loader'] is not None:
        #Evaluation on Eval Set(s) -->
        if configs['gan_mode']=='WGAN':
            losses_ed_eval,g_losses_eval,d_tru_err_eval,d_false_err_eval,i,src_eval_loader = test_adv_recons_loader(Models=Models_src,configs=configs,loader=configs['Source_Eval_Loader'],frm='Src',recons_mode=recons_mode)
        elif configs['gan_mode']=='GAN':
            losses_ed_eval,g_losses_eval,d_losses_eval,d_tru_err_eval,d_false_err_eval,i,src_eval_loader = test_adv_recons_loader(Models=Models_src,configs=configs,loader=configs['Source_Eval_Loader'],frm='Src',recons_mode=recons_mode)
            d_losses += d_losses_eval    
        g_losses+=g_losses_eval
        d_tru_err+=d_tru_err_eval
        d_false_err+=d_false_err_eval
        losses_ed_src+=losses_ed_eval
        j+=(i+1)

    if configs['Target_Eval_Loader'] is not None:
        if configs['gan_mode']=='WGAN':
            losses_ed_eval,g_losses_eval,d_tru_err_eval,d_false_err_eval,i,tar_eval_loader = test_adv_recons_loader(Models=Models_tar,configs=configs,loader=configs['Target_Eval_Loader'],frm='Tar',ret_loader=ret_loaders,recons_mode=recons_mode)
        elif configs['gan_mode']=='GAN':
            losses_ed_eval,g_losses_eval,d_losses_eval,d_tru_err_eval,d_false_err_eval,i,tar_eval_loader = test_adv_recons_loader(Models=Models_tar,configs=configs,loader=configs['Target_Eval_Loader'],frm='Tar',ret_loader=ret_loaders,recons_mode=recons_mode)
            d_losses += d_losses_eval
        g_losses+=g_losses_eval
        d_tru_err+=d_tru_err_eval
        d_false_err+=d_false_err_eval
        losses_ed_tar+=losses_ed_eval
        j+=(i+1)
    else:
        if ret_loaders:
            tar_eval_loader=None

    #Evaluation on Test Set(s) -->
    if configs['gan_mode']=='WGAN':
        losses_ed_eval,g_losses_eval,d_tru_err_eval,d_false_err_eval,i,src_test_loader = test_adv_recons_loader(Models=Models_src,configs=configs,loader=configs['Source_Test_Loader'],frm='Src',recons_mode=recons_mode)
    elif configs['gan_mode']=='GAN':
        losses_ed_eval,g_losses_eval,d_losses_eval,d_tru_err_eval,d_false_err_eval,i,src_test_loader = test_adv_recons_loader(Models=Models_src,configs=configs,loader=configs['Source_Test_Loader'],frm='Src',recons_mode=recons_mode)
        d_losses += d_losses_eval
    g_losses+=g_losses_eval
    d_tru_err+=d_tru_err_eval
    d_false_err+=d_false_err_eval
    losses_ed_src+=losses_ed_eval
    j+=(i+1)

    if configs['gan_mode']=='WGAN':
        losses_ed_eval,g_losses_eval,d_tru_err_eval,d_false_err_eval,i,tar_test_loader = test_adv_recons_loader(Models=Models_tar,configs=configs,loader=configs['Target_Test_Loader'],frm='Tar',ret_loader=ret_loaders,recons_mode=recons_mode)
    elif configs['gan_mode']=='GAN':
        losses_ed_eval,g_losses_eval,d_losses_eval,d_tru_err_eval,d_false_err_eval,i,tar_test_loader = test_adv_recons_loader(Models=Models_tar,configs=configs,loader=configs['Target_Test_Loader'],frm='Tar',ret_loader=ret_loaders,recons_mode=recons_mode)
        d_losses += d_losses_eval
    g_losses+=g_losses_eval
    d_tru_err+=d_tru_err_eval
    d_false_err+=d_false_err_eval
    losses_ed_tar+=losses_ed_eval
    j+=(i+1)

    logger['Test']['Gen_Loss'].append(g_losses/j)
    if configs['gan_mode']=='WGAN':
        logger['Test']['Disc_Loss'].append((d_tru_err+d_false_err)/j)
    elif configs['gan_mode']=='GAN':
        logger['Test']['Disc_Loss'].append((d_losses)/j)
    logger['Test']['Disc_Tru_Loss'].append(d_tru_err/j)
    logger['Test']['Disc_Fake_Loss'].append(d_false_err/j)
    if recons_mode == 'Target':
        logger['Test']['Recons_Loss_tar'].append(losses_ed_tar/j)

    return logger,src_eval_loader,tar_eval_loader,src_test_loader,tar_test_loader

# %%
def test_ae_loader(Models,configs,loader,ret_loader=False,to_shfl=False):
    Encoder = Models[0]
    Decoder = Models[1]
    Decoder.eval()
    Encoder.eval()

    if ret_loader:
        flag=True

    losses_ed=0
    with torch.no_grad():
        
        for i,(data_tuple) in enumerate(loader):
            x = data_tuple[0].to(device)
            if ret_loader:
                label = data_tuple[1]

            criterion = configs['criterion_recons']
            # PHASE1: Reconstruction Loss -->
            # if (just_recons or (frm=='Src' and configs['recons_domain'] in ['Source','Both']) or (frm=='Tar' and configs['recons_domain'] in ['Target','Both'])):
            if configs['recons_window']:
                x_ = data_tuple[-2].to(device)
                z = Encoder(x_)
                out = Decoder(z)
                y_ = data_tuple[-1].to(device)
                y_ = torch.squeeze(y_,dim=-1)
                loss = criterion(out, y_)
            else:
                z = Encoder(x)
                out = Decoder(z)
                loss = criterion(out, x)
            losses_ed += (loss.item())
        
            if ret_loader:
                if flag:
                    flag=False
                    batches = z
                    labels = label
                else:
                    batches = torch.cat((batches,z),dim=0)
                    labels = torch.cat((labels,label),dim=0)

    if not (ret_loader):
        return losses_ed,i,None
    else:
        data_set = TensorDataset(batches,labels)
        data_loader = DataLoader(dataset=data_set, batch_size=configs['batch_size'], shuffle=to_shfl)
        return losses_ed,i,data_loader

# %%
def test_ae_src_batch(Models,configs,logger,ret_loaders=False): # Used with Double Encoder Approach only
    losses_ed = 0
    j=0

    if configs['Source_Eval_Loader'] is not None:
        #Evaluation on Eval Set(s) -->
        losses_ed_eval,i,src_eval_loader = test_ae_loader(Models=Models,configs=configs,loader=configs['Source_Eval_Loader'],ret_loader=ret_loaders)
        losses_ed+=losses_ed_eval
        j+=(i+1)
    else:
        if ret_loaders:
            src_eval_loader=None

    #Evaluation on Test Set(s) -->
    losses_ed_eval,i,src_test_loader = test_ae_loader(Models=Models,configs=configs,loader=configs['Source_Test_Loader'],ret_loader=ret_loaders)
    losses_ed+=losses_ed_eval
    j+=(i+1)

    logger['Test']['Recons_Loss_src'].append(losses_ed/j)

    return logger,src_eval_loader,src_test_loader