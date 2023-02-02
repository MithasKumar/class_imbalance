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

import utilis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# ************* Random Seed to all the devices ****************
def rand_seed_devices(configs):    
    torch.backends.cudnn.deterministic = configs['only_deterministic']
    torch.manual_seed(configs['rand_seed'])
    np.random.seed(configs['rand_seed'])
    random.seed(configs['rand_seed'])

# **************************************************** Train Utilities SitaRam ********************************************************
# %%
def train_encoder_decoder_batch(Encoder,Decoder,X,configs,is_tar,encoder_step=True): # Backprop through Dec(Enc(X))
    # Reconstruction Loss -->
    Encoder.train()
    Decoder.train()

    if encoder_step:
        Encoder.zero_grad()
    Decoder.zero_grad()

    criterion = configs['criterion_recons']
    if is_tar:
        opt_E = configs['optimizer_encoder_tar']
        opt_D = configs['optimizer_decoder_tar']
    else:
        opt_E = configs['optimizer_encoder_src']
        opt_D = configs['optimizer_decoder_src']

    loss=0
    z = Encoder(X)
    out = Decoder(z)
    loss += criterion(out,X)
    
    loss.backward()
    opt_D.step()
    if encoder_step:
        opt_E.step()

    return Encoder,Decoder,loss.item()

# %%
def train_wgan_disc_batch(Discriminator,Generator,source_x,target_x,configs,double_gen=False): # Update Discriminator in W-GAN Style
    r"""
    double_gen: bool => If using 2 generators; like we do in Double Encoder Approach; One generating real examples and the other one generating fake examples
    """
    if double_gen:
        Generator_fake = Generator[1]
        Generator_true = Generator[0]
        Generator_true.eval()
        Generator_fake.eval()
    else:
        Generator.eval() 
    Discriminator.train()

    critic_iters = configs['critic_iters']
    opt_Dis = configs['optimizer_critic']

    d_tru_err=d_false_err=0
    if configs['wgan_train_mode']=='GP':
        d_pen_err=0
    d_losses = 0

    j=0
    while j<critic_iters:
        Discriminator.zero_grad()

        if double_gen:
            z_real = Variable(Generator_true(source_x).detach().data,requires_grad=False)
        else:
            z_real = Variable(Generator(source_x).detach().data,requires_grad=False)
        out_true = Discriminator(z_real)
        err_D_tru = configs['adv_pen']*torch.mean(out_true)
        
        if configs['min_max']:
            err_D_tru.backward(configs['mone'])
            err_D_tru*=-1
        else:
            err_D_tru.backward(configs['one'])
        d_tru_err+=err_D_tru

        if double_gen:
            z_fake = Variable(Generator_fake(target_x).detach().data,requires_grad=False)
        else:
            z_fake = Variable(Generator(target_x).detach().data,requires_grad=False)
        out_fake = Discriminator(z_fake)
        err_D_fake = configs['adv_pen']*torch.mean(out_fake)
        
        if configs['min_max']:
            err_D_fake.backward(configs['one'])
        else:
            err_D_fake.backward(configs['mone'])
            err_D_fake*=-1
        d_false_err+=err_D_fake

        if configs['wgan_train_mode']=='GP':
            err_D_penalty = configs['adv_pen']*configs['gp_penalty']*utilis.compute_gradient_penalty(Discriminator, z_real.data, z_fake.data)
            err_D_penalty.backward(configs['one'])
            d_pen_err+=err_D_penalty
        
        loss_d = err_D_tru + err_D_fake
        if configs['wgan_train_mode']=='GP':
            loss_d += err_D_penalty
        d_losses+=loss_d

        opt_Dis.step()

        if configs['wgan_train_mode']=='Clip':
            # Clip weights of discriminator
            for p in Discriminator.parameters():
                p.data.clamp_(configs['clip_lower'], configs['clip_upper'])

        j+=1

    if configs['wgan_train_mode']=='GP':
        return Discriminator,d_false_err.item()/critic_iters,d_tru_err.item()/critic_iters,d_pen_err.item()/critic_iters,d_losses.item()/critic_iters
    else:
        return Discriminator,d_false_err.item()/critic_iters,d_tru_err.item()/critic_iters,d_losses.item()/critic_iters

# %%
def train_gan_disc_batch(Discriminator,Generator,source_x,target_x,configs,double_gen=False): # Update Discriminator in OG GAN Style
    if double_gen:
        Generator_fake = Generator[1]
        Generator_true = Generator[0]
        Generator_true.eval()
        Generator_fake.eval()
    else:
        Generator.eval() 
    Discriminator.train()

    critic_iters = configs['critic_iters']
    opt_Dis = configs['optimizer_critic']
    smooth_label = configs['gan_smooth_label']
    criterion = nn.BCELoss().to(device)

    d_tru=d_fake=0
    d_losses = 0

    y_true = smooth_label*torch.ones((source_x.size(0), 1)).to(device)
    y_fake = torch.zeros((source_x.size(0), 1)).to(device)

    j=0
    while j<critic_iters:
        Discriminator.zero_grad()

        if double_gen:
            z_real = Variable(Generator_true(source_x).detach().data,requires_grad=False)
        else:
            z_real = Variable(Generator(source_x).detach().data,requires_grad=False)
        out_true = Discriminator(z_real)
        d_tru += torch.mean(out_true)

        if double_gen:
            z_fake = Variable(Generator_fake(target_x).detach().data,requires_grad=False)
        else:
            z_fake = Variable(Generator(target_x).detach().data,requires_grad=False)
        out_fake = Discriminator(z_fake)
        d_fake += torch.mean(out_fake)
        
        loss_d = configs['adv_pen']*(criterion(out_true, y_true) + criterion(out_fake, y_fake))
        loss_d.backward()

        d_losses+=loss_d
        opt_Dis.step()

        j+=1

    return Discriminator,d_fake.item()/critic_iters,d_tru.item()/critic_iters,d_losses.item()/critic_iters

# %%
def train_generator_batch(Discriminator,Generator,target_x,configs,encoder_step=True): # Update Generator Weights
    Generator.train() # Switch on the Dropouts
    Discriminator.eval()

    opt_G = configs['optimizer_generator']
    gen_iters_sr = configs['gen_iters']
    if not encoder_step:
        gen_iters_sr=1
    
    if encoder_step:
        Generator.zero_grad()

    j=0
    while j<gen_iters_sr:
        z_fake = Generator(target_x) 
        out_fake = Discriminator(z_fake)

        if configs['gan_mode']=='GAN':
            smooth_label = configs['gan_smooth_label']
            y_true = smooth_label*torch.ones((target_x.size(0), 1)).to(device)
            criterion = nn.BCELoss().to(device)
                
        if configs['gan_mode']=='WGAN':
            loss_g = configs['adv_pen']*torch.mean(out_fake)

            if configs['min_max']:
                loss_g.backward(configs['mone'])
                loss_g *= -1
            else:
                loss_g.backward(configs['one'])

        elif configs['gan_mode']=='GAN':
            loss_g = configs['adv_pen']*criterion(out_fake,y_true)    
            loss_g.backward()

        if encoder_step:
            opt_G.step()
        j+=1

    return Generator,loss_g.item()

# %%
def train_classifier_batch(Model,X,y,configs,is_tar=False,Encoder=None,to_floatify=True): # Backprop error thru Classifier 
    r"""
    Model => Classifier; Encoder => If not None; X is projected to latent space first through the Encoder and then passed through Classifier
    to_floatify: bool => Whether to convert LongTensor type labels to float or not; generally done while using BCE Loss
    """
    Model.train()
    if Encoder is not None:
        Encoder.eval()
        X = Variable(Encoder(X).detach().data,requires_grad=False)

    criterion = configs['criterion_classify']
    if is_tar:
        opt = configs['optimizer_classifier_tar']
    else:
        opt = configs['optimizer_classifier']

    if configs['is_ensemble']:
        y = y.repeat_interleave(configs['ensemble_models'],1)

    if to_floatify:
        y = y.float()
    
    if is_tar:
        outs = Model(X,out_mode='Target')
    else:
        outs = Model(X)
    loss_cls = criterion(outs, y)

    # Backprop and perform optimisation
    opt.zero_grad()
    loss_cls.backward()
    opt.step()

    return Model,loss_cls

# %%
def train_classifier_encoder_batch(Model,Encoder,X,y,configs,encoder_step=True,mode='Classify',to_floatify=True): # Backprop through Classifier(Encoder(x)) or Window_Classifier(Encoder(x))
    r"""
    Model => Classifier (mode = Classify) Or Decoder which is a Window Classifier (mode = Window_src/Window_tar)
    encoder_step: bool => Whether to update Encoder params or not   
    to_floatify: bool => Whether to convert LongTensor type labels to float or not; generally done while using BCE Loss
    """
    Model.train()
    Encoder.train()
        
    Model.zero_grad()
    if encoder_step:
        Encoder.zero_grad()

    cls_pen=1

    if mode=='Classify':
        if to_floatify:
            y = y.to(torch.int64)#cuda().to(torch.float64)#type(torch.cuda.FloatTensor)
        cls_pen=configs['cls_pen']
        criterion = configs['criterion_classify']
        opt = configs['optimizer_classifier']
        if encoder_step:
            opt_cls = configs['optimizer_encoder_classifier']
    elif mode in ['Window_src','Window_tar']:
        y=torch.squeeze(y,dim=-1)
        dom=mode.split('_')[1]
        criterion = configs['criterion_recons']
        opt = configs['optimizer_decoder_'+dom]
        if encoder_step:
            opt_cls = configs['optimizer_encoder_'+dom]
    else:
        raise NotImplementedError("Unable to decipher mode in train_classifier_encoder_batch :(")

    z_real = Encoder(X)
    outs = Model(z_real)
    if configs['is_ensemble'] and mode=='Classify':
        y = y.repeat_interleave(configs['ensemble_models'],1)

    if mode in ['Window_src', 'Window_tar']:
        loss_cls = cls_pen * F.cross_entropy(outs,y)
    else:
        loss_cls = cls_pen*criterion(outs, y)
    loss_cls.backward()

    opt.step()
    if encoder_step:
        opt_cls.step()

    return Model,Encoder,loss_cls


def train_classifier_encoder_batch_autobalance(Model, Encoder, X, y, configs, encoder_step=True, mode='Classify',
                                   to_floatify=True):  # Backprop through Classifier(Encoder(x)) or Window_Classifier(Encoder(x))
    r"""
    Model => Classifier (mode = Classify) Or Decoder which is a Window Classifier (mode = Window_src/Window_tar)
    encoder_step: bool => Whether to update Encoder params or not
    to_floatify: bool => Whether to convert LongTensor type labels to float or not; generally done while using BCE Loss
    """
    Model.train()
    Encoder.train()

    Model.zero_grad()
    if encoder_step:
        Encoder.zero_grad()

    cls_pen = 1

    if mode == 'Classify':
        if to_floatify:
            y = y.float()
        cls_pen = configs['cls_pen']
        criterion = configs['criterion_classify']
        opt = configs['optimizer_classifier']
        if encoder_step:
            opt_cls = configs['optimizer_encoder_classifier']
    elif mode in ['Window_src', 'Window_tar']:
        y = torch.squeeze(y, dim=-1)
        dom = mode.split('_')[1]
        criterion = configs['criterion_recons']
        opt = configs['optimizer_decoder_' + dom]
        if encoder_step:
            opt_cls = configs['optimizer_encoder_' + dom]
    else:
        raise NotImplementedError("Unable to decipher mode in train_classifier_encoder_batch :(")

    z_real = Encoder(X)
    outs = Model(z_real)
    if configs['is_ensemble'] and mode == 'Classify':
        y = y.repeat_interleave(configs['ensemble_models'], 1)

    loss_cls = cls_pen * criterion(outs, y)
    loss_cls.backward()

    opt.step()
    if encoder_step:
        opt_cls.step()

    return Model, Encoder, loss_cls
