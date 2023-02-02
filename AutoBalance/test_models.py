# %%
# *********************** Importing Essential Libraries *************************
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import random
import numpy as np

import utilis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# ************* Random Seed to all the devices ****************
def rand_seed_devices(configs):    
    torch.backends.cudnn.deterministic = configs['only_deterministic']
    torch.manual_seed(configs['rand_seed'])
    np.random.seed(configs['rand_seed'])
    random.seed(configs['rand_seed'])

# %%
class MLP_model(nn.Module):
    def __init__(self,f1,f2,f3,ld):
        super(MLP_model, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.latent_dim = ld
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim,self.f1),
            nn.ReLU(),
            nn.Dropout(0.25))
        self.fc2 = nn.Sequential(
            nn.Linear(self.f1,self.f2),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(self.f2,self.f3),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc4 = nn.Linear(self.f3,1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return torch.sigmoid(out)

# %%
class MLP_ensemble(nn.Module):
	def __init__(self,F1,F2,F3,ld):
		super(MLP_ensemble, self).__init__()
		self.F1 = F1
		self.F2 = F2
		self.F3 = F3
		self.N = len(self.F1)
		self.ld = ld
		self.classifiers = nn.ModuleList()
		for k in range(self.N):
			self.classifiers.append(MLP_model(f1=self.F1[k],f2=self.F2[k],f3=self.F3[k],ld=self.ld))

	def forward(self, x):
		flag=True
		for classifier in self.classifiers:
			if flag:
				flag=False
				out = classifier(x)
			else:
				out = torch.cat((out,classifier(x)),dim=1)
		return out

# %%
class Discriminator(nn.Module):
	def __init__(self,ld,f1d,f2d,f3d,is_wgan=True):
		super(Discriminator, self).__init__()
		self.latent_dim = ld
		self.is_wgan = is_wgan
		self.f1d = f1d
		self.f2d = f2d
		self.f3d = f3d
		self.fc1 = nn.Sequential(
			nn.Linear(self.latent_dim,self.f1d),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(self.f1d,self.f2d),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(self.f2d,self.f3d),
			nn.ReLU())
		self.fc4 = nn.Linear(self.f3d,1)

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		if self.is_wgan:
			return out
		else:
			return torch.sigmoid(out)

# %%
class SN_Discriminator(nn.Module):
	def __init__(self,ld,f1d,f2d,f3d):
		super(SN_Discriminator, self).__init__()
		self.latent_dim = ld
		self.f1d = f1d
		self.f2d = f2d
		self.f3d = f3d
		self.fc1 = nn.Sequential(
			spectral_norm(nn.Linear(self.latent_dim,self.f1d)),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			spectral_norm(nn.Linear(self.f1d,self.f2d)),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			spectral_norm(nn.Linear(self.f2d,self.f3d)),
			nn.ReLU())
		self.fc4 = spectral_norm(nn.Linear(self.f3d,1))

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		out = self.fc4(out)
		return torch.sigmoid(out)

# %%
class Decoder_LSTM(nn.Module):
	def __init__(self, input_dim, n_features,N,tseq_len,apply_sig,to_squeeze=False):
		super(Decoder_LSTM, self).__init__()
		self.tsql = tseq_len
		self.apply_sig = apply_sig
		self.to_squeeze=to_squeeze
		self.Dec_Lyrs = N
		self.input_dim = input_dim
		self.hidden_dim, self.n_features = 2 * input_dim, n_features
		self.pre_lstms = nn.ModuleList()
		for k in range(self.Dec_Lyrs-1):
			self.pre_lstms.append(nn.LSTM(input_size=self.input_dim, hidden_size=self.input_dim, num_layers=1, batch_first=True))
		self.rnn_end = nn.LSTM(
			input_size=self.input_dim,
			hidden_size=self.hidden_dim,
			num_layers=1,
			batch_first=True
			)
		self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

	def forward(self, x):
		d1,d2 = x.size(0),x.size(1)
		#         print(x.shape)
		x = x.repeat_interleave(self.tsql,0).view(d1,self.tsql,d2)
		#         print(x.shape)
		for lyr in self.pre_lstms:
			x, (_, _) = lyr(x)
		x, (hidden_n, cell_n) = self.rnn_end(x)
		#         print(x.shape)
		outs = self.output_layer(x)
		if self.to_squeeze:
			outs = torch.squeeze(outs,dim=-1)
		if self.apply_sig:
			return torch.sigmoid(outs)
		else:
			return outs

# %%
class Encoder_LSTM(nn.Module):
	def __init__(self, n_features, embedding_dim,N,to_unsqueeze=False):
		super(Encoder_LSTM, self).__init__()
		self.n_features = n_features
		self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
		self.Enc_Lyrs = N
		self.to_unsqueeze=to_unsqueeze
		self.rnn_start = nn.LSTM(
			input_size=self.n_features,
			hidden_size=self.hidden_dim,
			num_layers=1,
			batch_first=True
			)
		self.hidden_lstms = nn.ModuleList()
		for k in range(self.Enc_Lyrs-2):
			self.hidden_lstms.append(nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True))
		self.rnn_end = nn.LSTM(
			input_size=self.hidden_dim,
			hidden_size=self.embedding_dim,
			num_layers=1,
			batch_first=True
			)

	def forward(self, x):
		tbs = x.size(0)
		if self.to_unsqueeze:
			x=torch.unsqueeze(x,dim=-1)
		x, (_, _) = self.rnn_start(x)
		for lyr in self.hidden_lstms:
			x, (_, _) = lyr(x)
		x, (hidden_n, _) = self.rnn_end(x)
		return hidden_n.view(tbs,-1)

# class Encoder(nn.Module):   # Old Version; Recommend NOT to USE this! 
#     def __init__(self, n_features, embedding_dim):
#         super(Encoder, self).__init__()
#         self.n_features = n_features
#         self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
#         self.rnn1 = nn.LSTM(
#           input_size=self.n_features,
#           hidden_size=self.hidden_dim,
#           num_layers=1,
#           batch_first=True
#         )
#         self.rnn2 = nn.LSTM(
#           input_size=self.hidden_dim,
#           hidden_size=self.hidden_dim,
#           num_layers=1,
#           batch_first=True
#         )
#         self.rnn3 = nn.LSTM(
#           input_size=self.hidden_dim,
#           hidden_size=self.embedding_dim,
#           num_layers=1,
#           batch_first=True
#         )
#     def forward(self, x):
#         tbs = x.size(0)
# #         print(x.shape)
#         x, (_, _) = self.rnn1(x)
# #         print(x.shape)
#         x, (_, _) = self.rnn2(x)
#         x, (hidden_n, _) = self.rnn3(x)
# #         print(hidden_n.shape)
#         return hidden_n.view(tbs,-1)
# #         return torch.squeeze(x) 

# %%
class classifier_combined(nn.Module):
	def __init__(self,f1,f2,f3,ld):
		super(classifier_combined, self).__init__()
		self.f1 = f1
		self.f2 = f2
		self.f3 = f3
		self.latent_dim = ld
		self.fc1 = nn.Sequential(
			nn.Linear(self.latent_dim,self.f1),
			nn.ReLU(),
			nn.Dropout(0.25))
		self.fc2 = nn.Sequential(
			nn.Linear(self.f1,self.f2),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.fc3 = nn.Sequential(
			nn.Linear(self.f2,self.f3),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.fc4 = nn.Linear(self.f3,1)
		self.fc5 = nn.Linear(self.f3,1)

	def forward(self, x, out_mode='Source'):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		if out_mode=='Source':
			out = self.fc4(out)
		elif out_mode=='Target':
			out = self.fc5(out)
		return torch.sigmoid(out)

# %%
class MLP_model_multi_outs(nn.Module):
	def __init__(self,N,nuerons_info,ld,apply_softmax=False):
		super(MLP_model_multi_outs, self).__init__()
		self.apply_softmax = apply_softmax
		self.lyrs = N
		self.F = nuerons_info[:-1]
		self.nc = nuerons_info[-1]
		self.latent_dim = ld
		if self.lyrs==1:
			self.F = [self.latent_dim]
		if self.lyrs>1:
			self.fc_start = nn.Sequential(
				nn.Linear(self.latent_dim,self.F[0]),
				nn.ReLU(),
				nn.Dropout(0.25))
			self.mid_layers = nn.ModuleList()
			for k in range(self.lyrs-2):
				self.mid_layers.append(nn.Sequential(nn.Linear(self.F[k],self.F[k+1]),nn.ReLU(),nn.Dropout(0.5)))
		self.fc_end = nn.Linear(self.F[-1],self.nc)

	def forward(self, x):
		if self.lyrs>1:
			out = self.fc_start(x)
			for lyr in self.mid_layers:
				out = lyr(out)
			out = self.fc_end(out)
		else:
			out = self.fc_end(x)
		if self.apply_softmax:
			return nn.Softmax()(out)
		else:
			return out

# %%
class Encoder_MLP(nn.Module):
	def __init__(self,nf,N,F,ld):
		super(Encoder_MLP, self).__init__()
		self.nf = nf
		self.F = F
		self.N = N
		self.latent_dim = ld
		self.fc_start = nn.Sequential(nn.Linear(self.nf,self.F[0]),nn.ReLU(),nn.Dropout(0.25))
		self.mid_layers = nn.ModuleList()
		for k in range(self.N-2):
			self.mid_layers.append(nn.Sequential(nn.Linear(self.F[k],self.F[k+1]),nn.ReLU(),nn.Dropout(0.5)))
		self.fc_end = nn.Linear(self.F[-1],self.latent_dim)

	def forward(self, x):
		out = self.fc_start(x)
		for lyr in self.mid_layers:
			out = lyr(out)
		out = self.fc_end(out)
		return out

# %%
class Decoder_MLP(nn.Module):
	def __init__(self,nf,N,F,ld,rev_lst,apply_sig):
		super(Decoder_MLP, self).__init__()
		if rev_lst:
			F.reverse()
		self.nf = nf
		self.F = F
		self.N = N
		self.latent_dim = ld
		self.apply_sig=apply_sig
		dropout_prob=0.5
		if self.N==2:
			dropout_prob=0.25
		self.fc_start = nn.Sequential(nn.Linear(self.latent_dim,self.F[0]),nn.ReLU(),nn.Dropout(dropout_prob))
		self.mid_layers = nn.ModuleList()
		for k in range(self.N-2):
			dropout_prob=0.5
			if k==self.N-3:
				dropout_prob=0.25
			self.mid_layers.append(nn.Sequential(nn.Linear(self.F[k],self.F[k+1]),nn.ReLU(),nn.Dropout(dropout_prob)))
		self.fc_end = nn.Linear(self.F[-1],self.nf)

	def forward(self, x):
		out = self.fc_start(x)
		for lyr in self.mid_layers:
			out = lyr(out)
		out = self.fc_end(out)
		if self.apply_sig:
			return torch.sigmoid(out)
		else:
			return out

# %%
class Decoder_1D_CNN(nn.Module):
	def __init__(self, embedding_dim, tseq_len, apply_sig, in_channels=1, to_unsqueeze=True):
		super().__init__()
		self.n_in = in_channels
		self.apply_sig = apply_sig
		self.to_unsqueeze = to_unsqueeze
		self.ld = embedding_dim
		self.tsql = tseq_len
		self.layer1_channels=16
		self.layer2_channels = self.layer1_channels*2
		self.layer3_channels = 1
		self.pads_kers_dils = utilis.calc_pads_kers_dils(seg_len=self.tsql)
		
		self.layer3 = nn.Sequential(
			nn.ConvTranspose1d(self.layer2_channels, self.layer3_channels, (self.pads_kers_dils[-1][1],), stride=2, padding=self.pads_kers_dils[-1][0], dilation=self.pads_kers_dils[-1][-1],output_padding=1),
			nn.BatchNorm1d(self.layer3_channels),
			nn.ReLU(),
			nn.Dropout(p=0.5)
			)

		self.layer2 = nn.Sequential(
			nn.ConvTranspose1d(self.layer1_channels, self.layer2_channels, (self.pads_kers_dils[1][1],), stride=2, padding=self.pads_kers_dils[1][0], dilation=self.pads_kers_dils[1][-1],output_padding=1),
			nn.BatchNorm1d(self.layer2_channels),
			nn.ReLU(),
			nn.Dropout(p=0.5)
			)

		self.layer1 = nn.Sequential(
			nn.ConvTranspose1d(self.n_in, self.layer1_channels, (self.pads_kers_dils[0][1],), stride=2, padding=self.pads_kers_dils[0][0], dilation=self.pads_kers_dils[0][-1], output_padding=1),
			nn.BatchNorm1d(self.layer1_channels),
			nn.ReLU(),
			nn.Dropout(p=0.5)
			)

		self.linear1 = nn.Linear(self.ld, self.tsql//8)

	def forward(self, x):
		if self.to_unsqueeze:
			x = torch.unsqueeze(x,dim=1)
		x = self.linear1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		if self.to_unsqueeze:
			x = torch.squeeze(x,dim=1)
		if self.apply_sig:
			return torch.sigmoid(x)
		else:
			return x

# %%
class Encoder_1D_CNN(nn.Module):
    def __init__(self, embedding_dim, tseq_len, in_channels=1, to_unsqueeze=True):
        super().__init__()
        self.n_in = in_channels
        self.to_unsqueeze = to_unsqueeze
        self.ld = embedding_dim
        self.tsql = tseq_len
        self.layer1_channels=32
        self.layer2_channels = self.layer1_channels//2
        self.layer3_channels = 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.n_in, self.layer1_channels, (9,), stride=1, padding=4),
            nn.BatchNorm1d(self.layer1_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.layer1_channels, self.layer2_channels, (5,), stride=1, padding=2),
            nn.BatchNorm1d(self.layer2_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(self.layer2_channels, self.layer3_channels, (5,), stride=1, padding=2),
            nn.BatchNorm1d(self.layer3_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2, stride=2)
        )
        self.linear1 = nn.Linear(self.tsql//8, self.ld)

    def forward(self, x):
        if self.to_unsqueeze:
            x = torch.unsqueeze(x,dim=1)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        if self.to_unsqueeze:
            x = torch.squeeze(x,dim=1)
        # print(x.shape)
        # print("")
        return x
