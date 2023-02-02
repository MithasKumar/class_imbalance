# %%
# *********************** Importing Essential Libraries *************************
import math
import numpy as np
import pandas as pd
import os
import torch
import time
import random
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
# import torch.utils.data 
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# ***************************** Random Seed to all the devices ************************************
def rand_seed_devices(configs):    
    torch.backends.cudnn.deterministic = configs['only_deterministic']
    torch.manual_seed(configs['rand_seed'])
    np.random.seed(configs['rand_seed'])
    random.seed(configs['rand_seed'])

# *************************** Data Processing Related Utilities *********************
# %%
def Intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

# %%
def send_random_indices(N,k,is_perc=False):
    if is_perc:
        k = int(0.01*k*N)
    if k<1:
        k = int(k*N)
    res_sr=[]
    for i in range(k):
        res_sr.append(random.randint(0,N-1))
    return res_sr

# %%
def get_subset_tensors(X,y,rat,supp_arrs=[]):
    N = len(y)
    k = int(rat*N)
    ones = sum(y)
    zeros = N-ones
    zeros_rat = zeros/N
    zeros_sub = int(k*zeros_rat)
    ones_sub = k-zeros_sub
    y_sr = torch.Tensor(y).type(torch.LongTensor)
    ones_idx = torch.where(y_sr==1)
    zeros_idx = torch.where(y_sr==0)
    X_ones_sr = X[ones_idx]
    X_zeros_sr = X[zeros_idx]
    y_ones_sr = y_sr[ones_idx]
    y_zeros_sr = y_sr[zeros_idx]
    if len(supp_arrs)>0:
        split_supp_ones_arrs=[]
        split_supp_zeros_arrs=[]
        for arr in supp_arrs:
            split_supp_ones_arrs.append(arr[ones_idx])
            split_supp_zeros_arrs.append(arr[zeros_idx])
    ones_sub_idx = send_random_indices(N=ones,k=ones_sub)
    zeros_sub_idx = send_random_indices(N=zeros,k=zeros_sub)
    X_sub = torch.cat((X_ones_sr[ones_sub_idx],X_zeros_sr[zeros_sub_idx]),dim=0)
    y_sub = torch.cat((y_ones_sr[ones_sub_idx],y_zeros_sr[zeros_sub_idx]),dim=0)
    res_supp_arrs=[]
    if len(supp_arrs)>0:
        for (arr_ones_sr,arr_zeros_sr) in zip(split_supp_ones_arrs,split_supp_zeros_arrs):
            res_supp_arrs.append(torch.cat((arr_ones_sr[ones_sub_idx],arr_zeros_sr[zeros_sub_idx]),dim=0))
    return X_sub,y_sub.tolist(),res_supp_arrs,zeros_sub,ones_sub

# %%
def Invert_Dict(org_dict):
    return {v: k for k, v in org_dict.items()}

# %%
def select_subset_df_sr(main_df_sr,ids_df_sr,batch_id_col_nm_sr):
    dfs_sr=[]
    ids_sr = list(ids_df_sr[batch_id_col_nm_sr])
    for id_sr in ids_sr:
        temp_sr = main_df_sr[main_df_sr[batch_id_col_nm_sr]==id_sr]
        dfs_sr.append(temp_sr)
    return pd.concat(dfs_sr,axis=0,ignore_index=True)

# %%
def get_zero_std_cols(data):
    sig = data.std()
    non_zero_cols = sig != 0
    return non_zero_cols[non_zero_cols].index.tolist()

# %%
def get_scaler(data, scaling):
    if scaling == "Min_Max":
        scaler = MinMaxScaler()
    elif scaling == "Standard_Scalar":
        scaler = StandardScaler()
    try:
        scaler.fit(data)
    except:
        # In case there is error, dont fit the scaler
        scaler.fit(numpy.array([2000]*(len(data)+10)).reshape(-1, 1))
    return scaler

# %%
def get_sampler(sampler,rand_seed):
    if sampler == "SMOTE":
        return SMOTE(random_state=rand_seed)
    elif sampler == "SMOTEENN":
        return SMOTEENN(random_state=rand_seed)

# %%
def col_mean_std(data):
    mu,sig = data.mean(),data.std()
    non_zero_cols_sr = sig != 0
    return non_zero_cols_sr,mu,sig

# %%
def col_min_max(data):
    return data.min(),data.max()

# %%
def uniform_norm(data,means,stds,nzc_sr):
    return (data.loc[:,nzc_sr]-means.loc[nzc_sr])/stds.loc[nzc_sr]

# %%
def min_max_norm(data,mins,maxs):
    return ((data-mins)/(maxs-mins)).fillna(1)

# %%
def repair_temp_cols(data,q1,q2,norm_type,domain,temp_thresh):
    if domain=='Source':
        search_key = 'Temp'
    elif domain=='Target':
        search_key = 'TC'
    temp_cols = []
    tot_cols = list(data.columns)
    for col_nm in tot_cols:      # Collecting temperature columns
        if search_key==col_nm.split('_')[0] or col_nm=='Temperature':
            temp_cols.append(col_nm)
    # print("")
    # print(temp_cols)
    # print("")
    for temp_col in temp_cols:
        curr_col_sr = data[data[temp_col]<temp_thresh][temp_col]
        if norm_type=='Standard_Scalar':
            _,x1,x2 = col_mean_std(data=curr_col_sr)
        elif norm_type=='Min_Max':
            x1,x2 = col_min_max(data=curr_col_sr)
        q1[temp_col] = x1
        q2[temp_col] = x2
    return q1,q2

# %%
def ret_chans_min_max_sr(x):
    channel_min = x.min(dim=1, keepdim=True)[0]
    channel_max = x.max(dim=1, keepdim=True)[0]
    return channel_min,channel_max

# %%
def norm_min_max_sr(x): # channel wise
    c1,c2 = ret_chans_min_max_sr(x)
    deno_sr = c2-c1
    deno_sr = torch.where(deno_sr==0,torch.Tensor([1]),deno_sr)
    return (x-c1)/deno_sr

# %%
def append_colnms_sr(data,nm):
    if data is not None:
        # return None
        cols_lst = data.columns
        new_cols_lst = [i+'_'+nm for i in cols_lst]
        data.columns = new_cols_lst

# %%
def make_window_str_to_dict(str_lst): # info_dict --> {window_set : 'Source/Target/Both', window_mode: 'Zeros/Noise', window_num: int}
    res_sr={}
    temp_lst = str_lst.split('_')
    for i in range(len(temp_lst)):
        if i==1:
            res_sr['window_mode'] = temp_lst[i]
        elif i==2:
            res_sr['window_num'] = int(temp_lst[i])
    return res_sr

# ************************** Downstream Task Related Utilities ******************************
# %%
def make_neurons_list_wc(str_lst,N):
    if N==1:
        return []
    temp_lst=str_lst[1:-1].split(',')
    fill_def_sr=False
    if len(temp_lst)<N-1:
        def_val=int(temp_lst[0])
        fill_def_sr=True
    res_sr=[]
    for i in range(N-1):
        if fill_def_sr:
            res_sr.append(def_val)
        else:
            res_sr.append(int(temp_lst[i]))
    return res_sr

# ***************** For loading-pretrained weights & freezing/un-freezing layers & resetting weights SitaRam *****************
# %%
def load_pretrain_enc_weights(Encoder,Encoder_pt,load_first_lyr,unfreeze_new_only):
    to_load_dict_sr = Encoder_pt.state_dict()
    flag=False
    try:
        if not load_first_lyr:
            flag=True
            model_dict_sr = Encoder.state_dict()
            for k in to_load_dict_sr.keys():
                if ('rnn_start' in k) or ('rnn1' in k) or ('fc_start' in k) or ('layer1' in k):
                    to_load_dict_sr[k] = model_dict_sr[k]
        Encoder.load_state_dict(to_load_dict_sr)
    except:
        flag=True
        model_dict_sr = Encoder.state_dict()
        for k in to_load_dict_sr.keys():
            if ('rnn_start' in k) or ('rnn1' in k) or ('fc_start' in k) or ('layer1' in k): 
                to_load_dict_sr[k] = model_dict_sr[k]
        Encoder.load_state_dict(to_load_dict_sr)
    if flag and unfreeze_new_only:
        Encoder,trainable_params_lst_sr = freeze_layers(Model=Encoder,not_freeze='rnn_start')
        if len(trainable_params_lst_sr)==0:
            Encoder,trainable_params_lst_sr = freeze_layers(Model=Encoder,not_freeze='rnn1')
        if len(trainable_params_lst_sr)==0:
            Encoder,trainable_params_lst_sr = freeze_layers(Model=Encoder,not_freeze='fc_start')
        if len(trainable_params_lst_sr)==0:
            Encoder,trainable_params_lst_sr = freeze_layers(Model=Encoder,not_freeze='layer1')
        return Encoder,trainable_params_lst_sr
    return Encoder,None

# %%
def load_pretrain_cls_tar(configs,Cls_src,Cls_tar,lr_sr,wd_sr):
    pre_trained_dict_sr = Cls_src.state_dict()
    model_dict_sr = Cls_tar.state_dict()
    model_dict_sr.update(pre_trained_dict_sr)
    for k in model_dict_sr.keys():
        if 'fc5' in k:
            k_ = k.replace('fc5','fc4')
            model_dict_sr[k] = pre_trained_dict_sr[k_]
    Cls_tar.load_state_dict(model_dict_sr)
    Cls_tar,trainable_params = freeze_layers(Model=Cls_tar,not_freeze='fc5')
    configs['optimizer_classifier_tar'] = torch.optim.Adam(trainable_params,lr=lr_sr,weight_decay=wd_sr)
    return Cls_tar,configs

# %%
def freeze_layers(Model,not_freeze=None):
    trainable_params = []
    for name,param in Model.named_parameters():
        if not_freeze!=None and not_freeze in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    return Model,trainable_params

# %%
def unfreeze_lyrs(Model,configs,model_nm):
    maybe_opts_sr = create_possible_opt_names(mn=model_nm)
    if configs['unfreeze_first_lyr_only'] and configs['curr_epoch']>=configs['unfreeze_epochs']: # Checking whether to unfreeze now or not
        for name,param in Model.named_parameters():
            if not param.requires_grad:
                param.requires_grad=True
                for nm in maybe_opts_sr:
                    if nm in configs['opt_lst']:
                        opt_sr = configs[nm]
                        opt_sr.add_param_group({'params':param})
                        configs[nm] = opt_sr
    return configs,Model

# %%
def create_possible_opt_names(mn): # Create a list of Possible number of optimizers associated with a model
    if mn=='Encoder_src':
        return ['optimizer_encoder_src','optimizer_encoder_classifier']
    elif mn=='Encoder_tar':
        return ['optimizer_generator','optimizer_encoder_tar']
    elif mn=='Classifier_tar':
        return ['optimizer_classifier_tar']
    elif mn=='Classifier':
        return ['optimizer_classifier']
    elif mn=='Discriminator':
        return ['optimizer_critic']
    elif mn=='Decoder_src':
        return ['optimizer_decoder_src']
    elif mn=='Decoder_tar':
        return ['optimizer_decoder_tar']

# %%
def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m,nn.ConvTranspose1d) or isinstance(m, nn.LSTM):
        m.reset_parameters()

# ************************** For Fetching Optimizer(s) and Criterion(s) ***************************
# %%
def get_optimizer(model, opt_name,lr,wd,b1=None,b2=None,params_lst=None):
    if params_lst!=None:
        trainable_params=params_lst
    else:
        trainable_params=model.parameters()
    if opt_name=='Adam':
        optimizer = torch.optim.Adam(trainable_params, 
                    lr=lr,
                    weight_decay=wd,
                    betas=(b1,b2)
                    )
    elif opt_name=='RMSprop':
        optimizer = torch.optim.RMSprop(trainable_params, 
                    lr=lr,
                    weight_decay=wd,
                    )
    elif opt_name=='Adagrad':
        optimizer = torch.optim.Adagrad(trainable_params, 
                    lr=lr,
                    weight_decay=wd,
                    )
    return optimizer

# %%
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.75, gamma=2,is_ensemble=False,ensemble_models=None):
        super(WeightedFocalLoss, self).__init__()
        self.N = ensemble_models
        self.is_ensemble = is_ensemble
        if self.is_ensemble:
            self.alpha = torch.tensor([alpha, 1-alpha]).repeat(self.N,1).permute(1,0).to(device)
        else:
            self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        targets = targets.type(torch.long)
        if not self.is_ensemble:
            at = self.alpha.gather(0, targets.data.view(-1))
        else:
            at = self.alpha.gather(0, targets.data)
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# %%
def get_loss_criterion(criterion_nm,wfl_alpha=None,wfl_gamma=None,is_ensemble=None,ensemble_models=None):
    if criterion_nm==None:
        return None
    if criterion_nm=='BCE':
        criterion = nn.BCELoss().to(device)
    elif criterion_nm=='CE':
        criterion = nn.CrossEntropyLoss().to(device)
    elif criterion_nm=='MSE':
        criterion = nn.MSELoss().to(device)
    elif criterion_nm=='WFL':
        criterion = WeightedFocalLoss(
            alpha=wfl_alpha,
            gamma=wfl_gamma,
            is_ensemble=is_ensemble,
            ensemble_models=ensemble_models
        )
    return criterion

# %%
def add_criterions(configs):
    configs['criterion_classify'] = get_loss_criterion(criterion_nm=configs['classifier_loss_func'],wfl_alpha=configs['wfl_alpha'],wfl_gamma=configs['wfl_gamma'],is_ensemble=configs['is_ensemble'],ensemble_models=configs['ensemble_models'])
    if configs['recons_window']:
        configs['criterion_recons'] = get_loss_criterion(criterion_nm='CE')
    else:
        configs['criterion_recons'] = get_loss_criterion(criterion_nm=configs['recons_loss_func'])
    return configs

# ******************************** For Learning Rate Scheduling *****************************
# %%
def schdl_lr(epoch,eta_min,eta_max,max_epoch,ad_fac,epoch_beg): 
    epoch-=epoch_beg
    max_epoch-=epoch_beg
    if 0 <= epoch <= max_epoch:
        return eta_min + (0.5*(eta_max-eta_min)*(1+(ad_fac*math.cos(math.pi*(epoch/max_epoch)))))
    else:
        if ad_fac==-1: # Ascend
            if epoch<0:
                return eta_min
            else:
                return eta_max
        else:
            if epoch<0:
                return eta_max
            else:
                return eta_min

# %%
def opt_step(configs,opt_nm):
    opt = configs[opt_nm]
    ref_dict = configs[opt_nm+'_scheduler']
    if ref_dict['to_schedule']:
        ad_fac=-1
        if ref_dict['to_descend']:
            ad_fac=1
        new_lr_sr = schdl_lr(epoch=configs['curr_epoch'],eta_min=ref_dict['min_lr'],eta_max=ref_dict['max_lr'],max_epoch=ref_dict['max_epoch'],ad_fac=ad_fac,epoch_beg=ref_dict['min_epoch'])
        for g in opt.param_groups:
            g['lr'] = new_lr_sr
        if configs['prnt_lr_updates']:
            print("Learning Rate for "+opt_nm+" at epoch = "+str(configs['curr_epoch']+1)+" = "+str(opt.param_groups[0]['lr']))
        configs[opt_nm] = opt

# %%
def opt_steps(configs): # For changing optimizer LRs, if any LR Scheduling done.
    print("")
    for opt_nm in configs['opt_lst']:
        opt_step(configs=configs,opt_nm=opt_nm)

# %%
def update_scheduler(configs,opt_nm,str_lst,lr_info): # str_lst (contains scheduling info) = [eta_min,eta_max,ad_fac,min_epoch,max_epoch]; lr_info contains the def values
    schdl_dict={'to_schedule':False}
    if str_lst!=None:
        schdl_dict['to_schedule']=True
        temp_lst = str_lst.split(',')
        i=0
        while i<5:
            ele = temp_lst[i]
            if i==0:
                if ele=='[':
                    schdl_dict['min_lr'] = lr_info[opt_nm]
                else:
                    schdl_dict['min_lr'] = float(ele[1:])
            elif i==1:
                if ele=='':
                    schdl_dict['max_lr'] = lr_info[opt_nm]
                else:
                    schdl_dict['max_lr'] = float(ele)
            elif i==2:
                schdl_dict['to_descend'] = ele=='Descend'
            elif i==3:
                schdl_dict['min_epoch']=int(ele)    
            elif i==4:
                schdl_dict['max_epoch']=int(ele[:-1])    
            i+=1
    configs[opt_nm+'_scheduler']=schdl_dict

# ********************************* For Target Subsampling *************************************
# %%
def make_subset_dict(string):
    if string is None:
        return None
    rat = float(string)
    if rat>1:
        rat/=100
    return {"ratio":rat}

# ******************************* For threshold tuning for Target Data *******************************
# %%
def make_preds(lmbd,probs):
    return torch.where(probs>=lmbd,torch.Tensor([1]),torch.Tensor([0]))

# %%
def calc_tpr_fpr(y_tru,thresh,y_prob):
    y_pred = make_preds(lmbd=thresh,probs=y_prob)
    n = y_tru.shape[0]
    tot_pos = y_tru.sum().item()
    tot_neg = n-tot_pos
    tp = torch.where((y_tru==1)&(y_pred==1),torch.Tensor([1]),torch.Tensor([0])).sum().item()
    fp = torch.where((y_tru==0)&(y_pred==1),torch.Tensor([1]),torch.Tensor([0])).sum().item()
    return fp/tot_neg,tp/tot_pos

# ******************************** Sampler for Imbalanced Data SitaRam ***************************
# %%
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

# ******************************** For multi-outs classifier scores calculation ***********************
# %%
def calc_accuracy_frm_multi_class(y_preds,y_tru):
    y_preds=y_preds.cpu()
    y_tru=y_tru.cpu()
    # print(y_tru.shape,y_preds.shape)
    return 100*accuracy_score(y_true=y_tru, y_pred=y_preds)

# ******************************* For training a WGAN with GP **************************************
# %%
def compute_gradient_penalty(Discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = Discriminator(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ******************************* For making plots and running tsne ***********************
# %%
def plot_train_val_curve(train_loss_list,test_loss_list,no_of_epochs,nm,path,is_percent=True,mark_fifty=False):
    if test_loss_list!=[]:
        plt.plot(range(no_of_epochs),train_loss_list,label = 'Train '+nm)
        if len(test_loss_list)==2:
            plt.plot(range(no_of_epochs),test_loss_list[0],label = 'Val '+nm)
            plt.plot(range(no_of_epochs),test_loss_list[1],label = 'Test '+nm)
        else:
            plt.plot(range(no_of_epochs),test_loss_list,label = 'Val '+nm)
        plt.title('Train-Val '+nm+' Curve')
        plt.legend()
    else:
        plt.plot(range(no_of_epochs),train_loss_list)
        plt.title(nm+' Curve')
    plt.grid()
    if is_percent:
        plt.axis([-0.7,no_of_epochs+1,-0.5,100+0.5])
        if mark_fifty or 'AUC_Score' in nm:
            plt.hlines(y = 50, xmin = -0.7, xmax = no_of_epochs+1, linestyles='dotted' ,colors='r')
    plt.xlabel('No. of epochs --->')
    plt.ylabel(nm+' ---> ')
    plt.savefig(os.path.join(path,nm+'.png'))
    plt.clf()

# %%
def plotProbs(gt, pred, path, nm, N=2):
    predictions = pd.DataFrame({"proba": pred, "gt": gt})
    sns.histplot(data=predictions, x="proba", hue="gt", bins=100)
    plt.savefig(os.path.join(path,nm+'_Probab_Plot.png'))
    plt.clf()
    cases_sr = [i for i in range(N)]
    for case in cases_sr:
        predictions = pd.DataFrame({
            "proba": [pred[i] for i in range(len(pred)) if gt[i] == case],
            "gt": [gt[i] for i in range(len(gt)) if gt[i] == case]})
        sns.histplot(data=predictions, x="proba", hue="gt", bins=100)
        plt.savefig(os.path.join(path,nm+'_Class_'+str(case)+'_Probab_Plot.png'))
        plt.clf()

# %%
def plot_roc_curve(y_true, y_prob,nm,best_thresh,path='./results/'):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    best_thresh = round(thresholds[index], ndigits = 4)
    print(fpr)
    print('-------')
    print(tpr)
    print('-------')
    print(thresholds)
    # plot the roc curve for the model
    # sns.lineplot(fpr, tpr, marker='.')
    plt.plot(fpr,tpr)
    ix = np.where(thresholds==best_thresh)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    plt.grid()
    plt.xlabel('False Positive Rate -->')
    plt.ylabel('True Positive Rate -->')
    plt.title(f"Best Threshold = {best_thresh}")
    plt.legend()
    plt.savefig(os.path.join(path,nm+'_ROC_Curve.png'))
    plt.clf()
    return best_thresh

# %%
def plot_tsne(df,nm,col_nm,lgnd_stat,path,alph=0.7):
    plt.figure(figsize=(8,6))
    if nm is None:
        nm = 'TSNE'
        plt.title("t-SNE Plot")
    else:
        plt.title("t-SNE Plot using "+nm)
    sns.scatterplot(
        x="tsne-x-axis", y="tsne-y-axis",
        hue=col_nm,
        data=df,
        legend = lgnd_stat,
        alpha=alph
    )
    plt.grid()
    plt.savefig(os.path.join(path,nm+'.png'))
    plt.clf()

# %%
def run_tsne(data,key_wrd,want_lbls,N,verb_stat=1,perp=40,iters=301):
    feat_cols = [ 'feature'+str(i)+'_'+key_wrd for i in range(N) ]
    if verb_stat!=0:
        time_start = time.time()
    tsne = TSNE(n_components=2, verbose=verb_stat, perplexity=perp, n_iter=iters)
    tsne_results = tsne.fit_transform(data[feat_cols].dropna(axis='rows'))
    if verb_stat!=0:
        print('t-SNE on '+key_wrd+' Latent Features done! Time elapsed: {} seconds'.format(time.time()-time_start))
    temp_df = pd.DataFrame({'tsne-x-axis' : tsne_results[:,0], 'tsne-y-axis' : tsne_results[:,1]})
    if want_lbls:
        temp_df['lbls'] = [str(i)+'_'+key_wrd for i in data['y1_'+key_wrd].dropna(axis='rows')]
    else:
        temp_df['lbls'] = key_wrd
    return temp_df

# %%
def dissolve_df(data,split_col_nm='lbls'):  
    i,k = 0,0
    df1_sr,df2_sr = pd.DataFrame(columns=data.columns),pd.DataFrame(columns=data.columns)
    for j in range(data.shape[0]):
        temp_lbl = data[split_col_nm].loc[j]
        if temp_lbl.split('_')[-1]=='src':
            df1_sr.loc[i] = list(data.loc[j])
            i+=1
        else:
            df2_sr.loc[k] = list(data.loc[j])
            k+=1
    return df1_sr,df2_sr

# %%
def make_sp_tsne(data,sp_lst,main_data,key_wrd,path):
    for i in range(len(sp_lst)):
        data['sp'+str(i)] = list(main_data[sp_lst[i]+'_'+key_wrd].dropna(axis='rows').values)
        plot_tsne(df=data,nm=sp_lst[i],col_nm='sp'+str(i),lgnd_stat='full',path=path)

# %%
def make_tsne(configs):
    # Running t-SNE -->
    temp_df = run_tsne(data=configs['tsne_df'],key_wrd='src',want_lbls=configs['want_lbls'],N=configs['latent_dim'])
    print("")
    if configs['to_plt_tsne']=='Both':
        temp2_df = run_tsne(data=configs['tsne_df'] ,key_wrd='tar',want_lbls=configs['want_lbls'],N=configs['latent_dim'])
        temp_df = pd.concat([temp_df,temp2_df],axis=0,ignore_index=True)

    temp_plt_nm=None
    if configs['want_lbls']:
        temp_plt_nm="True_Labels"
    
    # Plotting t-SNE -->
    plot_tsne(df=temp_df,nm=temp_plt_nm,col_nm='lbls',lgnd_stat='full',path=configs['save_plot_path'])
    if configs['want_sp']:
        temp_df_src,temp_df_tar = dissolve_df(data=temp_df)
        make_sp_tsne(data=temp_df_src,sp_lst=configs['Source_configs']["setpoint_parameters"],main_data=configs['tsne_df'],key_wrd='src',path=configs['save_src_tsne_plots_path'])
        if configs['to_plt_tsne']=='Both':
            make_sp_tsne(data=temp_df_tar,sp_lst=configs['Target_configs']["setpoint_parameters"],main_data=configs['tsne_df'],key_wrd='tar',path=configs['save_tar_tsne_plots_path'])
