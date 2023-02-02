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

class TimeSeriesCnnModel(nn.Module):
	def __init__(self):
		super(TimeSeriesCnnModel, self).__init__()
		self.network = nn.Sequential(
			nn.Conv1d(19, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(negative_slope=0.02),
			nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(64),
			nn.LeakyReLU(negative_slope=0.02),
			nn.MaxPool1d(2),
			nn.Dropout(0.1),

			nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(negative_slope=0.02),
			nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(negative_slope=0.02),
			nn.MaxPool1d(2),
			nn.Dropout(0.1),

			nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(negative_slope=0.02),
			nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(negative_slope=0.02),
			nn.MaxPool1d(2),
			nn.Dropout(0.1),

			nn.Flatten(),
			nn.Linear(9472, 1024),
			nn.LeakyReLU(negative_slope=0.02),
			nn.Dropout(0.3),
			nn.Linear(1024, 1024),
			nn.LeakyReLU(negative_slope=0.02),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(negative_slope=0.02),
			nn.Dropout(0.3),
			nn.Linear(512, 2))

	def forward(self, xb):
		x = self.network(xb)
		return x

class ResNet(nn.Module):
	def __init__(self, n_in, n_classes):
		super(ResNet, self).__init__()
		self.n_in = n_in
		self.n_classes = n_classes

		blocks = [19, 64]
		self.blocks = nn.ModuleList()
		for b, _ in enumerate(blocks[:-1]):
			self.blocks.append(ResidualBlock(*blocks[b:b + 2], self.n_in))

		self.fc1 = nn.Linear(blocks[-1], self.n_classes)

	def forward(self, x: torch.Tensor):
		for block in self.blocks:
			x = block(x)
		x = F.avg_pool2d(x, 2)
		x = torch.mean(x, dim=2)
		x = x.view(-1, 1, 128)
		x = self.fc1(x)


		return x.view(-1, self.n_classes)


class ResidualBlock(nn.Module):
	def __init__(self, in_maps, out_maps, time_steps):
		super(ResidualBlock, self).__init__()
		self.in_maps = in_maps
		self.out_maps = out_maps
		self.time_steps = time_steps

		self.conv1 = nn.Conv2d(self.in_maps, self.out_maps, (7, 1), 1, (3, 0))
		self.bn1 = nn.BatchNorm2d(self.out_maps)

		self.conv2 = nn.Conv2d(self.out_maps, self.out_maps, (5, 1), 1, (2, 0))
		self.bn2 = nn.BatchNorm2d(self.out_maps)

		self.conv3 = nn.Conv2d(self.out_maps, self.out_maps, (3, 1), 1, (1, 0))
		self.bn3 = nn.BatchNorm2d(self.out_maps)

	def forward(self, x):
		x = x.view(-1, self.in_maps, self.time_steps, 1)
		x = F.relu(self.bn1(self.conv1(x)))
		inx = x
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)) + inx)

		return x


class MLP_model_old(nn.Module):
    def __init__(self,f1,f2,f3,ld):
        super(MLP_model_old, self).__init__()
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
        self.fc4 = nn.Linear(self.f3,2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


class LSTM_Classifier(nn.Module):
	def __init__(self, n_features, embedding_dim,N,to_unsqueeze=False):
		super(LSTM_Classifier, self).__init__()
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
		self.final_fc = nn.Linear(self.embedding_dim, 2)

	def forward(self, x):
		tbs = x.size(0)
		if self.to_unsqueeze:
			x=torch.unsqueeze(x,dim=-1)
		x, (_, _) = self.rnn_start(x)
		for lyr in self.hidden_lstms:
			x, (_, _) = lyr(x)
		x, (hidden_n, _) = self.rnn_end(x)

		out = hidden_n.view(tbs,-1)
		out = self.final_fc(out)

		return out.view(tbs, -1)

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
        ####################################################################################################################

        '''ResNet in PyTorch.
        For Pre-activation ResNet, see 'preact_resnet.py'.
        Reference:
        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, BN=True):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(
			in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
							   stride=1, padding=1, bias=False)
		if BN:
			self.bn1 = nn.BatchNorm2d(planes)
			self.bn2 = nn.BatchNorm2d(planes)
		else:
			self.bn1 = nn.Sequential()
			self.bn2 = nn.Sequential()

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			if BN:
				self.shortcut = nn.Sequential(
					nn.Conv2d(in_planes, self.expansion * planes,
							  kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(self.expansion * planes)
				)
			else:
				self.shortcut = nn.Sequential(
					nn.Conv2d(in_planes, self.expansion * planes,
							  kernel_size=1, stride=stride, bias=False)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10, in_planes=16, BN=True):
		super(ResNet, self).__init__()
		self.in_planes = in_planes
		self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
							   stride=1, padding=1, bias=False)
		if BN:
			self.bn1 = nn.BatchNorm2d(in_planes)
		else:
			self.bn1 = nn.Sequential()
		self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, BN=BN)
		self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, BN=BN)
		self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, BN=BN)
		self.avg_pool = nn.AvgPool2d(8)
		self.linear = nn.Linear(in_planes * 4, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride, BN):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, BN=BN))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		# out = self.layer4(out)
		out = self.avg_pool(out)
		feature = out.view(out.size(0), -1)
		logits = self.linear(feature)
		return logits

def ResNet20_1D(num_classes=10, BN=True):
	num_res_blocks = int((20 - 2) / 6)
	return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks], num_classes=num_classes, BN=BN)

def test():
	net = ResNet20_1D(BN=True)
	print(sum(p.numel() for p in net.parameters() if p.requires_grad))
	x = torch.randn(2, 3, 32, 32)
	y = net(x)
	print(y)
	print(y[0].size(), y[1].size())
