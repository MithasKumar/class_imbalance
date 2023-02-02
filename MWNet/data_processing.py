# %%
# *********************** Importing Essential Libraries *************************
import numpy as np
import pandas as pd
import random
import json
import os

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import utilis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# ************* Random Seed to all the devices ****************
def rand_seed_devices(configs):    
    torch.backends.cudnn.deterministic = configs['only_deterministic']
    torch.manual_seed(configs['rand_seed'])
    np.random.seed(configs['rand_seed'])
    random.seed(configs['rand_seed'])

# ************************************************************** Raw Data Related Processing functions *************************************************************************
# %%
def save_mean_std(data,configs,set_nm,update_no_of_features=True):
	domain_name_sr = set_nm.split('_')[0]
	main_data_sr = data[configs[domain_name_sr+'_configs']["feature_columns"]]
	non_zero_cols_sr,mu,sig = utilis.col_mean_std(data=main_data_sr)
	# print(non_zero_cols_sr,mu,sig)
	mu,sig = utilis.repair_temp_cols(data=main_data_sr,q1=mu,q2=sig,norm_type='Standard_Scalar',domain=domain_name_sr,temp_thresh=configs['temp_thresh'])
	# print(mu,sig)
	configs[set_nm+'_mean'] = mu
	configs[set_nm+'_std'] = sig
	configs[set_nm+'_nz_idxs'] = non_zero_cols_sr
	if update_no_of_features:
		no_of_features = int(non_zero_cols_sr.sum())
		configs[domain_name_sr+'_no_of_features'] = no_of_features
		print("No. of Non-Zero Features ("+domain_name_sr+" Data) = "+str(no_of_features))
	return configs

# %%
def save_min_max(data,configs,set_nm):
	domain_name_sr = set_nm.split('_')[0]
	main_data_sr = data[configs[domain_name_sr+'_configs']["feature_columns"]]
	min_sr,max_sr = utilis.col_min_max(data=main_data_sr)
	min_sr,max_sr = utilis.repair_temp_cols(data=main_data_sr,q1=min_sr,q2=max_sr,norm_type='Min_Max',domain=domain_name_sr,temp_thresh=configs['temp_thresh'])
	configs[set_nm+'_min'] = min_sr
	configs[set_nm+'_max'] = max_sr
	return configs

# %%
def get_raw_data_configs(configs):
	src_configs = json.load(open(configs['Source_configs_path']))
	src_no_of_features = len(src_configs["feature_columns"])
	print("No. of Features (Source Data) = "+str(src_no_of_features))
	src_no_of_sp_params = len(src_configs["setpoint_parameters"])
	print("No. of Set-point Parameters (Source Data) = "+str(src_no_of_sp_params))
	print("")
	tar_configs = json.load(open(configs['Target_configs_path']))
	tar_no_of_features = len(tar_configs["feature_columns"])
	print("No. of Features (Target Data) = "+str(tar_no_of_features))
	tar_no_of_sp_params = len(tar_configs["setpoint_parameters"])
	print("No. of Set-point Parameters (Target Data) = "+str(tar_no_of_sp_params))
	print("")

	configs['Target_configs'] = tar_configs
	configs['Target_no_of_features'] = tar_no_of_features
	configs['Target_no_of_sp_params'] = tar_no_of_sp_params

	configs['Source_configs'] = src_configs
	configs['Source_no_of_features'] = src_no_of_features
	configs['Source_no_of_sp_params'] = src_no_of_sp_params

	return configs

# %%
def get_raw_data_df(configs):
	configs = get_raw_data_configs(configs=configs)

	src_train_df_sr = pd.read_feather(configs['Source_train_data_path'])
	src_eval_df_sr = pd.read_feather(configs['Source_eval_data_path'])
	src_test_df_sr = pd.read_feather(configs['Source_test_data_path'])

	tar_train_df_sr = pd.read_feather(configs['Target_train_data_path'])
	tar_eval_df_sr = pd.read_feather(configs['Target_eval_data_path'])
	tar_test_df_sr = pd.read_feather(configs['Target_test_data_path'])

	# if configs['train_cls_tar']:
	# 	configs['Target_train_lbls_df'] = pd.read_feather(configs['Target_train_lbls_data_path'])
	# 	configs['Target_eval_lbls_df'] = pd.read_feather(configs['Target_eval_lbls_data_path'])

	configs['Source_train_df'] = src_train_df_sr
	configs['Source_eval_df'] = src_eval_df_sr
	configs['Source_test_df'] = src_test_df_sr

	configs['Target_train_df'] = tar_train_df_sr
	configs['Target_eval_df'] = tar_eval_df_sr
	configs['Target_test_df'] = tar_test_df_sr

	# if configs['data_norm_lvl']=='Global':
	# 	if configs['data_norm_mode']=='Standard_Scalar':
	# 		configs = save_mean_std(data=src_train_df_sr,configs=configs,set_nm='Source_train')
	# 		if not configs['impose_scalers']:
	# 			configs = save_mean_std(data=tar_train_df_sr,configs=configs,set_nm='Target_train')
	# 	elif configs['data_norm_mode']=='Min_Max':
	# 		configs = save_min_max(data=src_train_df_sr,configs=configs,set_nm='Source_train')
	# 		if not configs['impose_scalers']:
	# 			configs = save_min_max(data=tar_train_df_sr,configs=configs,set_nm='Target_train')
	# 	print("")

	return configs

# %%
def print_raw_data_df_info(configs):
	
	print("Source Data Info --> ")
	print("No. of wheels in train data = "+str(len(configs['Source_train_df'][configs['Source_configs']["batch_id_column"]].unique())))
	print("Product ids in train data = "+str(configs['Source_train_df'][configs['Source_configs']["product_id_column"]].unique()))
	print("No. of wheels in eval data = "+str(len(configs['Source_eval_df'][configs['Source_configs']["batch_id_column"]].unique())))
	print("Product ids in eval data = "+str(configs['Source_eval_df'][configs['Source_configs']["product_id_column"]].unique()))
	print("No. of wheels in test data = "+str(len(configs['Source_test_df'][configs['Source_configs']["batch_id_column"]].unique())))
	print("Product ids in test data = "+str(configs['Source_test_df'][configs['Source_configs']["product_id_column"]].unique()))

	print("Target Data Info --> ")
	print("No. of wheels in train data = "+str(len(configs['Target_train_df'][configs['Target_configs']["batch_id_column"]].unique())))
	print("Product ids in train data = "+str(configs['Target_train_df'][configs['Target_configs']["product_id_column"]].unique()))
	print("No. of wheels in eval data = "+str(len(configs['Target_eval_df'][configs['Target_configs']["batch_id_column"]].unique())))
	print("Product ids in eval data = "+str(configs['Target_eval_df'][configs['Target_configs']["product_id_column"]].unique()))
	print("No. of wheels in test data = "+str(len(configs['Target_test_df'][configs['Target_configs']["batch_id_column"]].unique())))
	print("Product ids in test data = "+str(configs['Target_test_df'][configs['Target_configs']["product_id_column"]].unique()))

# %%
def get_psuedo_window_label_tuple_sr(X,wm,wn,tsql):
	wind_len = tsql//wn
	X_new = X.detach().clone()
	ps_lbl_sr = random.randint(0,wn-1)
	if wm=='Zeros':
		X_new[:,(wind_len*ps_lbl_sr):(wind_len*(ps_lbl_sr+1)),:] = 0
	elif wm=='Noise':
		X_new[:,(wind_len*ps_lbl_sr):(wind_len*(ps_lbl_sr+1)),:] = torch.randn(X.shape[0],wind_len,X.shape[-1])
	return X_new,ps_lbl_sr

# %%
class Raw_Dataset(torch.utils.data.Dataset):
	r"""Dataset wrapping tensors.

	Each sample will be retrieved by indexing tensors along the first dimension.

	Args:
		*tensors (Tensor): tensors that have the same size of the first dimension.
	"""

	def __init__(self, data, targets, data2=None, targets2=None) -> None:

		self.data = data
		self.targets_list = targets
		self.tot_len = len(targets)
		self.targets = torch.Tensor(targets).type(torch.LongTensor).view(self.tot_len,1)
		self.data2=data2
		self.targets2=targets2
        
	def __getitem__(self, index):
		if self.data2 is None and self.targets2 is None:
			return (self.data[index], self.targets[index])
		elif self.data2 is not None and self.targets2 is None:
			return (self.data[index], self.targets[index], self.data2[index])
		elif self.data2 is not None and self.targets2 is not None:
			return (self.data[index], self.targets[index], self.data2[index], self.targets2[index])

	def __len__(self):
		return self.data.size(0)

	def get_labels(self):
		return self.targets_list

# %%
def get_raw_df_to_loader(data,configs,is_lbls,tseq_len,blen,norm_dict,ps_lbls_dict=None,prnt_data_dist=True,to_shfl=False,is_src=True,want_sampler=False,subset_dict=None):
	flag=False
	if ps_lbls_dict==None:
		ps_lbls_dict={}
		ps_lbls_dict['want_ps_lbls']=False
	want_ps_lbls=False
	if 	ps_lbls_dict['want_ps_lbls']:
		want_ps_lbls = (ps_lbls_dict['window_set'] in ['Source','Both'] and is_src) or (ps_lbls_dict['window_set'] in ['Target','Both'] and not is_src)
	if want_ps_lbls:
		ps_lbls=[]
		flag2=True
	if (is_lbls):
		lbls = []
	if prnt_data_dist:
		ones=zeros=0
	batch_groups = data.groupby(configs["batch_id_column"])
	for  bid, batch in batch_groups:
		btch = batch[configs["feature_columns"]]
		if btch.isna().sum().sum()==0:
			if norm_dict['data_norm_lvl']=='Global':
				if norm_dict['data_norm_mode']=='Standard_Scalar':
					if not is_src and not norm_dict['impose_scalers']:
						btch = utilis.uniform_norm(data=btch,means=norm_dict['Target_train_mean'],stds=norm_dict['Target_train_std'],nzc_sr=norm_dict['Target_train_nz_idxs'])
					else:
						btch = utilis.uniform_norm(data=btch,means=norm_dict['Source_train_mean'],stds=norm_dict['Source_train_std'],nzc_sr=norm_dict['Source_train_nz_idxs'])
				elif norm_dict['data_norm_mode']=='Min_Max':
					if not is_src and not norm_dict['impose_scalers']:
						btch = utilis.min_max_norm(data=btch,mins=norm_dict['Target_train_min'],maxs=norm_dict['Target_train_max'])
					else:
						btch = utilis.min_max_norm(data=btch,mins=norm_dict['Source_train_min'],maxs=norm_dict['Source_train_max'])
			elif norm_dict['data_norm_lvl']=='Local' and norm_dict['data_norm_mode']=='Standard_Scalar':
				non_zero_cols_sr,mu,sig = utilis.col_mean_std(data=btch)
				btch = utilis.uniform_norm(data=btch,means=mu,stds=sig,nzc_sr=non_zero_cols_sr)

			if prnt_data_dist or is_lbls:
				lbl = batch[configs["target_label"]].unique()[0]
				if (lbl!=0) and (lbl!=1):
					lbl=0
				if is_lbls:
					lbls.append(lbl)
                    
			btch = torch.from_numpy(btch.to_numpy()).float() # btch --> some seq_len,no_of_channels
			btch = torch.unsqueeze(torch.transpose(btch,0,1),dim=0)
			btch = torch.squeeze(nn.AdaptiveAvgPool1d(tseq_len)(btch),dim=0)
			out = torch.transpose(btch,0,1)
			out = torch.unsqueeze(out,dim=0)
			if norm_dict['data_norm_lvl']=='Local' and norm_dict['data_norm_mode']=='Min_Max':
				out = utilis.norm_min_max_sr(x=out)

			if flag==False:
				flag=True
				batches_sr = out
			else:
				batches_sr = torch.cat((batches_sr,out),dim=0)

			if prnt_data_dist:
				if lbl==1:
					ones+=1
				else:
					zeros+=1

			if want_ps_lbls:
				X_,y_ = get_psuedo_window_label_tuple_sr(X=out,wm=ps_lbls_dict['window_mode'],wn=ps_lbls_dict['window_num'],tsql=tseq_len)
				# print(X_.shape)
				if flag2:
					flag2=False
					batches2_sr = X_
				else:
					batches2_sr = torch.cat((batches2_sr,X_),dim=0)
				ps_lbls.append(y_)
    
	if prnt_data_dist and subset_dict is None:
		print("Number of 0s present in the dataset = "+str(zeros))
		print("Number of 1s present in the dataset = "+str(ones))
		if want_ps_lbls:
			np_lbls = np.asarray(ps_lbls)
			for i in range(ps_lbls_dict['window_num']):
				print("Number of "+str(i)+" psuedo labels present in the dataset = "+str(np.count_nonzero(np_lbls==i)))
    
	if is_lbls:
		if want_ps_lbls:
			tot_len = len(ps_lbls)
			ps_lbls = torch.Tensor(ps_lbls).type(torch.LongTensor).view(tot_len,1)
			if want_sampler:
				set_sr = Raw_Dataset(data=batches_sr,targets=lbls,data2=batches2_sr,targets2=ps_lbls)
			else:
				if subset_dict is not None:
					batches_sr,lbls,res_supp_arrs,zeros,ones = utilis.get_subset_tensors(X=batches_sr,y=lbls,rat=subset_dict["ratio"],supp_arrs=[batches2_sr,ps_lbls])
					batches2_sr,ps_lbls = res_supp_arrs[0],res_supp_arrs[1]
				tot_len = len(lbls)
				lbls = torch.Tensor(lbls).type(torch.LongTensor).view(tot_len,1)
				set_sr = TensorDataset(batches_sr,lbls,batches2_sr,ps_lbls)
		else:
			if want_sampler:
				set_sr = Raw_Dataset(data=batches_sr,targets=lbls)
			else:
				if subset_dict is not None:
					batches_sr,lbls,_,zeros,ones = utilis.get_subset_tensors(X=batches_sr,y=lbls,rat=subset_dict["ratio"])
				tot_len = len(lbls)
				lbls = torch.Tensor(lbls).type(torch.LongTensor).view(tot_len,1)
				set_sr = TensorDataset(batches_sr,lbls)

	else:
		if want_ps_lbls:
			tot_len = len(ps_lbls)
			ps_lbls = torch.Tensor(ps_lbls).type(torch.LongTensor).view(tot_len,1)
			set_sr = TensorDataset(batches_sr,batches2_sr,ps_lbls)
		else:
			set_sr = TensorDataset(batches_sr)
	if want_sampler and is_lbls:
		data_loader = DataLoader(dataset=set_sr, batch_size=blen, sampler=utilis.ImbalancedDatasetSampler(set_sr))
	else:
		data_loader = DataLoader(dataset=set_sr, batch_size=blen, shuffle=to_shfl)

	if prnt_data_dist and subset_dict is not None:
		print("Number of 0s present in the dataset = "+str(zeros))
		print("Number of 1s present in the dataset = "+str(ones))
		if want_ps_lbls:
			np_lbls = np.asarray(ps_lbls)
			for i in range(ps_lbls_dict['window_num']):
				print("Number of "+str(i)+" psuedo labels present in the dataset = "+str(np.count_nonzero(np_lbls==i)))

	if prnt_data_dist:
		return data_loader,zeros,ones, set_sr
	else:
		return data_loader

# %%
def make_ps_lbls_dict_sr(configs): # info_dict --> {window_set : 'Source/Target/Both', window_mode: 'Zeros/Noise', window_num: int}
	if configs['recons_window']:
		info_dict = utilis.make_window_str_to_dict(configs['win_recons_info'])
		info_dict["window_set"]=configs["recons_domain"]
		info_dict['want_ps_lbls']=True
		cls_info=utilis.make_neurons_list_wc(str_lst=configs['win_cls_configs'],N=configs['cls_lyrs'])
		cls_info.append(info_dict['window_num'])
		configs['window_classifer_info'] = cls_info
	else:
		info_dict={}
		info_dict['want_ps_lbls']=False

	return configs,info_dict

# %%
def get_raw_data_to_loaders(configs):

	norm_dict = {}
	norm_dict['data_norm_lvl'] = configs['data_norm_lvl']
	norm_dict['data_norm_mode'] = configs['data_norm_mode']
	norm_dict['impose_scalers'] = configs['impose_scalers']
	# if configs['data_norm_lvl']=='Global':
	# 	if configs['data_norm_mode']=='Standard_Scalar':	
	# 		norm_dict['Source_train_mean'] = configs['Source_train_mean']
	# 		norm_dict['Source_train_std'] = configs['Source_train_std']
	# 		norm_dict['Source_train_nz_idxs'] = configs['Source_train_nz_idxs']
	# 		if not norm_dict['impose_scalers']:
	# 			norm_dict['Target_train_mean'] = configs['Target_train_mean']
	# 			norm_dict['Target_train_std'] = configs['Target_train_std']
	# 			norm_dict['Target_train_nz_idxs'] = configs['Target_train_nz_idxs']
	# 	elif configs['data_norm_mode']=='Min_Max':
	# 		norm_dict['Source_train_min'] = configs['Source_train_min']
	# 		norm_dict['Source_train_max'] = configs['Source_train_max']
	# 		if not norm_dict['impose_scalers']:
	# 			norm_dict['Target_train_min'] = configs['Target_train_min']
	# 			norm_dict['Target_train_max'] = configs['Target_train_max']

	configs,psl_dict_sr = make_ps_lbls_dict_sr(configs=configs)

	print("")
	print("Making Source Train Loader --> ")
	configs['Source_Train_Loader'],configs['Source_Train_Negatives'],configs['Source_Train_Positives'], configs['Source_Train_Dataset'] = get_raw_df_to_loader(data=configs['Source_train_df'],configs=configs['Source_configs'],is_lbls=True,to_shfl=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],norm_dict=norm_dict,ps_lbls_dict=psl_dict_sr,want_sampler=configs['use_loader_sampler'])
	print("Making Source Eval Loader --> ")
	configs['Source_Eval_Loader'],configs['Source_Eval_Negatives'],configs['Source_Eval_Positives'], configs['Source_Eval_Dataset'] = get_raw_df_to_loader(data=configs['Source_eval_df'],configs=configs['Source_configs'],is_lbls=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],norm_dict=norm_dict,ps_lbls_dict=psl_dict_sr)
	print("Making Source Test Loader --> ")
	configs['Source_Test_Loader'],configs['Source_Test_Negatives'],configs['Source_Test_Positives'], configs['Source_Test_Dataset'] = get_raw_df_to_loader(data=configs['Source_test_df'],configs=configs['Source_configs'],is_lbls=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],norm_dict=norm_dict,ps_lbls_dict=psl_dict_sr)
	print("")
	print("Making Target Train Loader --> ")
	configs['Target_Train_Loader'],configs['Target_Train_Negatives'],configs['Target_Train_Positives'], configs['Target_Train_Dataset'] = get_raw_df_to_loader(data=configs['Target_train_df'],configs=configs['Target_configs'],is_lbls=True,to_shfl=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],is_src=False,norm_dict=norm_dict,ps_lbls_dict=psl_dict_sr,subset_dict=configs["subset_dict"])
	print("Making Target Eval Loader --> ")
	configs['Target_Eval_Loader'],configs['Target_Eval_Negatives'],configs['Target_Eval_Positives'], configs['Target_Eval_Dataset'] = get_raw_df_to_loader(data=configs['Target_eval_df'],configs=configs['Target_configs'],is_lbls=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],is_src=False,norm_dict=norm_dict,ps_lbls_dict=psl_dict_sr)
	print("Making Target Test Loader --> ")
	configs['Target_Test_Loader'],configs['Target_Test_Negatives'],configs['Target_Test_Positives'], configs['Target_Test_Dataset'] = get_raw_df_to_loader(data=configs['Target_test_df'],configs=configs['Target_configs'],is_lbls=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],is_src=False,norm_dict=norm_dict,ps_lbls_dict=psl_dict_sr)
	# print(configs['train_cls_tar'])
	# if configs['train_cls_tar']:
	# 	print("Making Target Train (with labels) Loader --> ")
	# 	configs['Target_Train_lbls_Loader'],configs['Target_Train_labels_Negatives'],configs['Target_Train_labels_Positives'] = get_raw_df_to_loader(data=configs['Target_train_lbls_df'],configs=configs['Target_configs'],is_lbls=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],is_src=False,norm_dict=norm_dict)
	# 	print("Making Target Test (with labels) Loader --> ")
	# 	configs['Target_Test_lbls_Loader'],configs['Target_Test_labels_Negatives'],configs['Target_Test_labels_Positives'] = get_raw_df_to_loader(data=configs['Target_eval_lbls_df'],configs=configs['Target_configs'],is_lbls=True,tseq_len=configs['adap_len'],blen=configs['batch_size'],is_src=False,norm_dict=norm_dict)

	return configs

# %%
def get_raw_df_to_embeds(data,Encoder,configs,tseq_len,blen,latent_dim,norm_dict,no_of_sp_params=None,prnt_lbl_dist=True,is_sp=False,to_shfl=False,to_df=True,save_bids=False,is_src=True,want_sampler=False):
	if Encoder is None:
		if prnt_lbl_dist and to_df:
			return None,None,None,None
		elif prnt_lbl_dist and not to_df:
			return None,None,None
		elif not prnt_lbl_dist and not to_df:
			return None
		else:
			return None,None
	flag=True
	if is_sp:
		f2=True
	Encoder.eval()
	lbls = []
	if save_bids:
		btch_ids_sr = []
	if prnt_lbl_dist:
		ones=zeros=0
	batch_groups = data.groupby(configs["batch_id_column"])
	for  bid , batch in batch_groups:
		btch = batch[configs["feature_columns"]]
		if btch.isna().sum().sum()==0:
			if save_bids:
				btch_ids_sr.append(bid)

			if norm_dict['data_norm_lvl']=='Global':
				if norm_dict['data_norm_mode']=='Standard_Scalar':
					if not is_src and not norm_dict['impose_scalers']:
						btch = utilis.uniform_norm(data=btch,means=norm_dict['Target_train_mean'],stds=norm_dict['Target_train_std'],nzc_sr=norm_dict['Target_train_nz_idxs'])
					else:
						btch = utilis.uniform_norm(data=btch,means=norm_dict['Source_train_mean'],stds=norm_dict['Source_train_std'],nzc_sr=norm_dict['Source_train_nz_idxs'])
				elif norm_dict['data_norm_mode']=='Min_Max':
					if not is_src and not norm_dict['impose_scalers']:
						btch = utilis.min_max_norm(data=btch,mins=norm_dict['Target_train_min'],maxs=norm_dict['Target_train_max'])
					else:
						btch = utilis.min_max_norm(data=btch,mins=norm_dict['Source_train_min'],maxs=norm_dict['Source_train_max'])
			elif norm_dict['data_norm_lvl']=='Local' and norm_dict['data_norm_mode']=='Standard_Scalar':
				non_zero_cols_sr,mu,sig = utilis.col_mean_std(data=btch)
				btch = utilis.uniform_norm(data=btch,means=mu,stds=sig,nzc_sr=non_zero_cols_sr)

			lbl = batch[configs["target_label"]].unique()[0]
			if (lbl!=0) and (lbl!=1):
				lbl=0
			lbls.append(lbl)

			btch = torch.from_numpy(btch.to_numpy()).float()
			btch = torch.unsqueeze(torch.transpose(btch,0,1),dim=0)
			btch = torch.squeeze(nn.AdaptiveAvgPool1d(tseq_len)(btch),dim=0)
			out = torch.transpose(btch,0,1)
			out = torch.unsqueeze(out,dim=0)

			if norm_dict['data_norm_lvl']=='Local' and norm_dict['data_norm_mode']=='Min_Max':
				whl = utilis.norm_min_max_sr(x=out).to(device)
			else:
				whl=out.to(device)
			with torch.no_grad():
				reps = Encoder(whl)

			if flag:
				flag=False
				res_sr = reps.cpu().numpy()
			else:
				res_sr = np.concatenate((res_sr,reps.cpu().numpy()),axis=0)

			if is_sp:
				btch = batch[configs["setpoint_parameters"]]
				if f2:
					f2=False
					sp_sr = np.reshape(btch.to_numpy()[0,:],(1,no_of_sp_params))
				else:
					sp_sr = np.concatenate((sp_sr,np.reshape(btch.to_numpy()[0,:],(1,no_of_sp_params))),axis=0)   

			if prnt_lbl_dist:
				if lbl==1:
					ones+=1
				else:
					zeros+=1
                
	if prnt_lbl_dist:
		print("Number of 0s present in the dataset = "+str(zeros))
		print("Number of 1s present in the dataset = "+str(ones))

	if want_sampler:
		set_sr = Raw_Dataset(data=torch.from_numpy(res_sr).float(),targets=lbls)
		loader = DataLoader(dataset=set_sr, batch_size=blen, shuffle=utilis.ImbalancedDatasetSampler(set_sr))
	else:
		tot_len = len(lbls)
		lbls = torch.Tensor(lbls).type(torch.LongTensor)
		set_sr = TensorDataset(torch.from_numpy(res_sr).float(),lbls.view(tot_len,1))
		loader = DataLoader(dataset=set_sr, batch_size=blen, shuffle=to_shfl)

	if to_df:        
		feat_cols = [ 'feature'+str(i) for i in range(latent_dim) ]
		df = pd.DataFrame(res_sr,columns=feat_cols)
		df['y1'] = lbls
		if save_bids:
			df[configs['batch_id_column']] = btch_ids_sr
		if is_sp:
			feat_cols = configs["setpoint_parameters"]
			# feat_cols = ['sp'+str(i) for i in range(no_of_sp_params)]
			df2 = pd.DataFrame(sp_sr,columns=feat_cols)
			df = pd.concat([df,df2],axis=1,ignore_index=False)
		print('Size of the dataframe: {}'.format(df.shape))
		if prnt_lbl_dist:
			return df,loader,zeros,ones
		else:
			return df,loader
	else:
		if prnt_lbl_dist:
			return loader,zeros,ones
		else:
			return loader

# %%
def get_raw_data_to_embed_loaders(configs,Encoder):

	norm_dict = {}
	norm_dict['data_norm_lvl'] = configs['data_norm_lvl']
	norm_dict['data_norm_mode'] = configs['data_norm_mode']
	norm_dict['impose_scalers'] = configs['impose_scalers']
	if configs['data_norm_lvl']=='Global':
		if configs['data_norm_mode']=='Standard_Scalar':	
			norm_dict['Source_train_mean'] = configs['Source_train_mean']
			norm_dict['Source_train_std'] = configs['Source_train_std']
			norm_dict['Source_train_nz_idxs'] = configs['Source_train_nz_idxs']
			if not norm_dict['impose_scalers']:
				norm_dict['Target_train_mean'] = configs['Target_train_mean']
				norm_dict['Target_train_std'] = configs['Target_train_std']
				norm_dict['Target_train_nz_idxs'] = configs['Target_train_nz_idxs']
		elif configs['data_norm_mode']=='Min_Max':
			norm_dict['Source_train_min'] = configs['Source_train_min']
			norm_dict['Source_train_max'] = configs['Source_train_max']
			if not norm_dict['impose_scalers']:
				norm_dict['Target_train_min'] = configs['Target_train_min']
				norm_dict['Target_train_max'] = configs['Target_train_max']

	Encoder_src = Encoder[0]
	Encoder_tar = Encoder[1]

	if configs['to_save_embeds'] or configs['to_plt_tsne']!='None':
		print("Making Train Reps Dataframe (Source) --> ")
		configs['Source_train_df_tsne'],configs['Source_Train_Loader'],configs['Source_Train_Negatives'],configs['Source_Train_Positives'] = get_raw_df_to_embeds(data=configs['Source_train_df'],configs=configs['Source_configs'],Encoder=Encoder_src,tseq_len=configs['adap_len'],is_sp=(configs['want_sp'] or configs['to_save_embeds']),to_shfl=True,save_bids=configs['to_save_embeds'],blen=configs['batch_size'],latent_dim=configs['latent_dim'],norm_dict=norm_dict,no_of_sp_params=configs['Source_no_of_sp_params'],want_sampler=False)
		print("")
		print("Making Eval Reps Dataframe (Source) --> ")
		configs['Source_eval_df_tsne'],configs['Source_Eval_Loader'],configs['Source_Eval_Negatives'],configs['Source_Eval_Positives'] = get_raw_df_to_embeds(data=configs['Source_eval_df'],configs=configs['Source_configs'],Encoder=Encoder_src,tseq_len=configs['adap_len'],is_sp=(configs['want_sp'] or configs['to_save_embeds']),save_bids=configs['to_save_embeds'],blen=configs['batch_size'],latent_dim=configs['latent_dim'],norm_dict=norm_dict,no_of_sp_params=configs['Source_no_of_sp_params'])
		print("")
		print("Making Test Reps Dataframe (Source) --> ")
		configs['Source_test_df_tsne'],configs['Source_Test_Loader'],configs['Source_Test_Negatives'],configs['Source_Test_Positives'] = get_raw_df_to_embeds(data=configs['Source_test_df'],configs=configs['Source_configs'],Encoder=Encoder_src,tseq_len=configs['adap_len'],is_sp=(configs['want_sp'] or configs['to_save_embeds']),save_bids=configs['to_save_embeds'],blen=configs['batch_size'],latent_dim=configs['latent_dim'],norm_dict=norm_dict,no_of_sp_params=configs['Source_no_of_sp_params'])
		print("")
		print("Making Train Reps Dataframe (Target) --> ")
		configs['Target_train_df_tsne'],configs['Target_Train_Loader'],configs['Target_Train_Negatives'],configs['Target_Train_Positives'] = get_raw_df_to_embeds(data=configs['Target_train_df'],configs=configs['Target_configs'],Encoder=Encoder_tar,tseq_len=configs['adap_len'],is_sp=(configs['want_sp'] or configs['to_save_embeds']),save_bids=configs['to_save_embeds'],blen=configs['batch_size'],latent_dim=configs['latent_dim'],is_src=False,norm_dict=norm_dict,no_of_sp_params=configs['Target_no_of_sp_params'])
		print("")
		print("Making Eval Reps Dataframe (Target) --> ")
		configs['Target_eval_df_tsne'],configs['Target_Eval_Loader'],configs['Target_Eval_Negatives'],configs['Target_Eval_Positives'] = get_raw_df_to_embeds(data=configs['Target_eval_df'],configs=configs['Target_configs'],Encoder=Encoder_tar,tseq_len=configs['adap_len'],is_sp=(configs['want_sp'] or configs['to_save_embeds']),save_bids=configs['to_save_embeds'],blen=configs['batch_size'],latent_dim=configs['latent_dim'],is_src=False,norm_dict=norm_dict,no_of_sp_params=configs['Target_no_of_sp_params'])
		print("")
		print("Making Test Reps Dataframe (Target) --> ")
		configs['Target_test_df_tsne'],configs['Target_Test_Loader'],configs['Target_Test_Negatives'],configs['Target_Test_Positives'] = get_raw_df_to_embeds(data=configs['Target_test_df'],configs=configs['Target_configs'],Encoder=Encoder_tar,tseq_len=configs['adap_len'],is_sp=(configs['want_sp'] or configs['to_save_embeds']),save_bids=configs['to_save_embeds'],blen=configs['batch_size'],latent_dim=configs['latent_dim'],is_src=False,norm_dict=norm_dict,no_of_sp_params=configs['Target_no_of_sp_params'])
		print("")

		utilis.append_colnms_sr(data=configs['Source_train_df_tsne'],nm='src')
		utilis.append_colnms_sr(data=configs['Source_eval_df_tsne'],nm='src')
		utilis.append_colnms_sr(data=configs['Source_test_df_tsne'],nm='src')
		utilis.append_colnms_sr(data=configs['Target_train_df_tsne'],nm='tar')
		utilis.append_colnms_sr(data=configs['Target_eval_df_tsne'],nm='tar')
		utilis.append_colnms_sr(data=configs['Target_test_df_tsne'],nm='tar')

		if configs['Target_train_df_tsne'] is None:
			configs['Train_tsne_df'] = configs['Source_train_df_tsne']
			configs['Test_tsne_df'] = configs['Source_eval_df_tsne']
			configs['Eval_tsne_df'] = configs['Source_test_df_tsne']
		else:
			configs['Train_tsne_df'] = pd.concat([configs['Source_train_df_tsne'],configs['Target_train_df_tsne']],axis=1,ignore_index=False)
			configs['Eval_tsne_df'] = pd.concat([configs['Source_eval_df_tsne'],configs['Target_eval_df_tsne']],axis=1,ignore_index=False)
			configs['Test_tsne_df'] = pd.concat([configs['Source_test_df_tsne'],configs['Target_test_df_tsne']],axis=1,ignore_index=False)

		if configs['to_save_embeds']:
			configs['Train_tsne_df'].to_csv(configs['save_train_embeds_path'],index=False)
			configs['Eval_tsne_df'].to_csv(configs['save_eval_embeds_path'],index=False)
			configs['Test_tsne_df'].to_csv(configs['save_test_embeds_path'],index=False)

		if configs['to_plt_tsne']!='None':
			configs['tsne_df'] = pd.concat([configs['Train_tsne_df'],configs['Eval_tsne_df'],configs['Test_tsne_df']],axis=0,ignore_index=True)
			print("Net df shape (for t-SNE Plots) = "+str(configs['tsne_df'].shape))

	else:

		print("Making Train loader (Source) --> ")
		configs['Source_Train_Loader'],configs['Source_Train_Negatives'],configs['Source_Train_Positives'] = get_raw_df_to_embeds(data=configs['Source_train_df'],configs=configs['Source_configs'],Encoder=Encoder_src,tseq_len=configs['adap_len'],to_shfl=True,to_df=False,blen=configs['batch_size'],latent_dim=configs['latent_dim'],norm_dict=norm_dict,want_sampler=False)
		print("")
		print("Making Eval loader (Source) --> ")
		configs['Source_Eval_Loader'],configs['Source_Eval_Negatives'],configs['Source_Eval_Positives'] = get_raw_df_to_embeds(data=configs['Source_eval_df'],configs=configs['Source_configs'],Encoder=Encoder_src,tseq_len=configs['adap_len'],to_df=False,blen=configs['batch_size'],latent_dim=configs['latent_dim'],norm_dict=norm_dict)
		print("")
		print("Making Test loader (Source) --> ")
		configs['Source_Test_Loader'],configs['Source_Test_Negatives'],configs['Source_Test_Positives'] = get_raw_df_to_embeds(data=configs['Source_test_df'],configs=configs['Source_configs'],Encoder=Encoder_src,tseq_len=configs['adap_len'],to_df=False,blen=configs['batch_size'],latent_dim=configs['latent_dim'],norm_dict=norm_dict)
		print("")
		print("Making Train loader (Target) --> ")
		configs['Target_Train_Loader'],configs['Target_Train_Negatives'],configs['Target_Train_Positives'] = get_raw_df_to_embeds(data=configs['Target_train_df'],configs=configs['Target_configs'],Encoder=Encoder_tar,tseq_len=configs['adap_len'],to_df=False,blen=configs['batch_size'],latent_dim=configs['latent_dim'],is_src=False,norm_dict=norm_dict)
		print("")
		print("Making Eval loader (Target) --> ")
		configs['Target_Eval_Loader'],configs['Target_Eval_Negatives'],configs['Target_Eval_Positives'] = get_raw_df_to_embeds(data=configs['Target_eval_df'],configs=configs['Target_configs'],Encoder=Encoder_tar,tseq_len=configs['adap_len'],to_df=False,blen=configs['batch_size'],latent_dim=configs['latent_dim'],is_src=False,norm_dict=norm_dict)
		print("")
		print("Making Test loader (Target) --> ")
		configs['Target_Test_Loader'],configs['Target_Test_Negatives'],configs['Target_Test_Positives'] = get_raw_df_to_embeds(data=configs['Target_test_df'],configs=configs['Target_configs'],Encoder=Encoder_tar,tseq_len=configs['adap_len'],to_df=False,blen=configs['batch_size'],latent_dim=configs['latent_dim'],is_src=False,norm_dict=norm_dict)
		print("")

	return configs

# ************************************************************** HCF Related Data Processing functions *************************************************************************
# %%
def make_psuedo_labels_df_sr(df_sr,psl_dict):
	df_copy_sr = df_sr.copy(deep=True)
	df_copy_sr.reset_index(drop=True)
	wm,wn=psl_dict['window_mode'],psl_dict['window_num']
	wind_len = df_sr.shape[-1]//wn
	ps_lbls = [random.randint(0,wn-1) for i in range(df_sr.shape[0])]

	for i,ps_lbl_sr in enumerate(ps_lbls):
		temp_arr = df_copy_sr.iloc[i,:].values
		# print(temp_arr.shape)
		if wm=='Zeros':
			temp_arr[(wind_len*ps_lbl_sr):(wind_len*(ps_lbl_sr+1))]=0 
		elif wm=='Noise':
			temp_arr[(wind_len*ps_lbl_sr):(wind_len*(ps_lbl_sr+1))]=np.random.rand(wind_len)
		# print(temp_arr)
		df_copy_sr.iloc[i,:] = temp_arr
	# print(df_copy_sr.shape)
	return df_copy_sr,pd.DataFrame(ps_lbls)

# %%
def transform_data(scaler, train, eval, test):
    train = pd.DataFrame(scaler.transform(train), columns=train.columns)
    eval = pd.DataFrame(scaler.transform(eval), columns=eval.columns)
    test = pd.DataFrame(scaler.transform(test), columns=test.columns)
    return (scaler, train, eval, test)

# %%
def print_psuedo_lbls_info_sr(lbls_df_sr,wn_sr):
	np_arr_sr = lbls_df_sr.values
	for i in range(wn_sr):
		print("Number of "+str(i)+" psuedo labels present in the dataset = "+str(np.count_nonzero(np_arr_sr==i)))

# %%
def add_pos_neg_counts_sr(tar_df_sr,keywrd,configs,psuedo_dict):
	domain=keywrd.split('_')[0]
	domain_set_sr=keywrd.split('_')[1]
	configs[keywrd+'_Positives'] = int(tar_df_sr.sum())
	configs[keywrd+'_Negatives'] = tar_df_sr.shape[0]-configs[keywrd+'_Positives']
	print("Making "+keywrd+" Loader --> ")
	print("Number of 0s present in the dataset = "+str(configs[keywrd+'_Negatives']))
	print("Number of 1s present in the dataset = "+str(configs[keywrd+'_Positives']))
	if psuedo_dict!=None and psuedo_dict[domain]!=None:
		print_psuedo_lbls_info_sr(lbls_df_sr=psuedo_dict[domain][domain_set_sr+'_lbls'],wn_sr=psuedo_dict['wn_sr'])
	print("")
	return configs

# %%
def data_df_to_embeds_df_sr(Encoder,df_sr,feat_cols):
	if Encoder is None:
		return None
	Encoder.eval()
	with torch.no_grad():
		out_np_arr_sr = Encoder(torch.from_numpy(df_sr.values).float().to(device)).cpu().numpy()
	return pd.DataFrame(out_np_arr_sr,columns=feat_cols)

# %%
def get_sp_df_sr(data,meta_df_sr,configs,domain):
	f2=True
	feat_cols = configs[domain+'_configs']["setpoint_parameters"]
	ids_sr = list(meta_df_sr[configs['Source_configs']["batch_id_column"]])
	for id_sr in ids_sr:
		btch = data[data[configs[domain+'_configs']["batch_id_column"]]==id_sr][feat_cols]
		no_of_sp_params = len(feat_cols)
		if btch.isna().sum().sum()==0 and btch.shape[0]>0:
			if f2:
				f2=False
				sp_sr = np.reshape(btch.to_numpy()[0,:],(1,no_of_sp_params))
			else:
				sp_sr = np.concatenate((sp_sr,np.reshape(btch.to_numpy()[0,:],(1,no_of_sp_params))),axis=0) 

	return pd.DataFrame(sp_sr,columns=feat_cols)

# %%
def make_df_for_hcf_tsne_sr(configs,domain,data,meta_df_sr,embeds_df_sr,lbls_df_sr):
	if embeds_df_sr is None:
		return None
	embeds_df_sr['y1'] = list(lbls_df_sr[configs['Source_configs']["target_label"]])        
	if configs['to_save_embeds']:
		embeds_df_sr[configs['Source_configs']['batch_id_column']] = list(meta_df_sr[configs['Source_configs']['batch_id_column']])
	if configs['want_sp'] or configs['to_save_embeds']:
		embeds_df_sr = pd.concat([embeds_df_sr,get_sp_df_sr(data=data,meta_df_sr=meta_df_sr,configs=configs,domain=domain)],axis=1,ignore_index=False)
	print('Size of the dataframe: {}'.format(embeds_df_sr.shape))
	return embeds_df_sr

# %%
class HCFDataset(torch.utils.data.Dataset):
	r"""Dataset wrapping tensors.

	Each sample will be retrieved by indexing tensors along the first dimension.

	Args:
		*tensors (Tensor): tensors that have the same size of the first dimension.
	"""

	def __init__(self, data_df, target_df, meta_df, psl_dict) -> None:

		if isinstance(data_df, pd.DataFrame):
			self.data = torch.from_numpy(data_df.values)
			self.target = torch.from_numpy(target_df.values)
		else:
			self.data = data_df
			self.target = target_df

		self.want_psl = False
		if psl_dict!=None:
			self.data2 = torch.from_numpy(psl_dict[psl_dict['key_wrd']+'_df_sr'].values)
			self.target2 = torch.from_numpy(psl_dict[psl_dict['key_wrd']+'_lbls'].values)
			self.want_psl = True

		self.valid_meta = False
		if isinstance(meta_df, list):
			self.meta = pd.concat([pd.DataFrame(df) for df in meta_df]) 
			self.valid_meta = True
		elif isinstance(meta_df, pd.DataFrame):
			self.meta = meta_df
			self.valid_meta = True

		if self.valid_meta:
			assert self.data.shape[0] == self.meta.shape[0] == self.target.shape[0], "Size mismatch between dfs"
		else:
			assert self.data.shape[0] == self.target.shape[0], "Size mismatch between data and labels"
        
	def __getitem__(self, index):
		if self.valid_meta:
			if self.want_psl:
				return (self.data[index].float(), self.target[index], self.meta.iloc[index].to_dict(), self.data2[index].float(), self.target2[index])
			else:
				return (self.data[index].float(), self.target[index], self.meta.iloc[index].to_dict())
		else:
			meta = {
			"timestamp": -1,
			"wheel_id": -1,
			"Domain": -1
			}
			if self.want_psl:
				return (self.data[index].float(), self.target[index], meta, self.data2[index].float(), self.target2[index])
			else:
				return (self.data[index].float(), self.target[index], meta)

	def __len__(self):
		return self.data.size(0)

# %%
def get_data_loader(data_df, target_df, batch_size, psl_dict, keywrd=None, meta_df=None, shuffle=False):
	if data_df is None:
		return None
	if psl_dict!=None:
		psl_dict['key_wrd'] = keywrd
	dataset = HCFDataset(data_df, target_df, meta_df, psl_dict=psl_dict)
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
	return loader

# %%
def get_hcf_dataloaders(configs, prnt_processing_info, load_common_cols=True, remove_zero_std_cols=True, encoder_sr=None, ps_lbls_dict=None):
	"""
	Loads Data Loaders from HCF Files
	"""
	source_root = target_root = configs['data_path']

	label_column = "Status"
	meta_columns = ["timestamp", "wheel_id"]

	load_tar_lbls_loaders = (encoder_sr is None and configs['train_cls_tar'])

	source_train = pd.read_csv(os.path.join(source_root,"BP09_2019_train_features.csv"),engine='python')
	source_eval = pd.read_csv(os.path.join(source_root,"BP09_2019_eval_features.csv"),engine='python')
	source_test = pd.read_csv(os.path.join(source_root,"BP09_2019_test_features.csv"),engine='python')

	source_train[label_column] = source_train[label_column].astype(int)
	source_eval[label_column] = source_eval[label_column].astype(int)
	source_test[label_column] = source_test[label_column].astype(int)

	target_train = pd.read_csv(os.path.join(target_root,"da_target_train_features.csv"),engine='python')
	target_eval = pd.read_csv(os.path.join(target_root,"da_target_eval_features.csv"),engine='python')
	target_test = pd.read_csv(os.path.join(target_root,"da_target_test_features.csv"),engine='python')

	if load_tar_lbls_loaders:
		target_train_lbls_ids_sr = pd.read_csv(os.path.join(target_root,'da_target_train_lbls_batch_ids.csv'),engine='python')
		target_train_lbls = utilis.select_subset_df_sr(main_df_sr=target_train,ids_df_sr=target_train_lbls_ids_sr,batch_id_col_nm_sr='FullSerialNo')
		target_eval_lbls_ids_sr = pd.read_csv(os.path.join(target_root,'da_target_eval_lbls_batch_ids.csv'),engine='python')
		target_eval_lbls = utilis.select_subset_df_sr(main_df_sr=target_eval,ids_df_sr=target_eval_lbls_ids_sr,batch_id_col_nm_sr='FullSerialNo')

    ##### Map Target Column Names to Source Ones - standardizing column names
	mapper = {}

	mapper["Label"] = label_column
	mapper["t_stamp"] = "timestamp"
	mapper["FullSerialNo"] = "wheel_id"

	target_air_flow_cols = [c for c in target_train.columns if "AirFlow_" in c]
	for c in target_air_flow_cols:
		mapper[c] = c.replace("AirFlow_", "Flowrate_Air_channel")
        
	temparature_mapper = {"TC_1" : "Temp_Channel1","TC_2" : "Temp_Channel2","TC_3" : "Temp_RootofSpoke","TC_4" : "Temp_SprueBush",}   
	target_temp_cols = [c for c in target_train.columns if "TC_" in c]
	for c in target_temp_cols:
		for key, value in temparature_mapper.items():
			if key in c:
				mapper[c] = c.replace(key, value)

	target_train.rename(mapper, axis=1, inplace=True)
	target_eval.rename(mapper, axis=1, inplace=True)
	target_test.rename(mapper, axis=1, inplace=True)
	if load_tar_lbls_loaders:
		target_train_lbls.rename(mapper, axis=1, inplace=True)
		target_eval_lbls.rename(mapper, axis=1, inplace=True)

	missing_columns = ['Presc_sp_Flowrate_Air_channel3',
	'Presc_sp_Flowrate_Air_channel4', 
	'Presc_sp_Flowrate_Air_channel5',
	'Presc_sp_Flowrate_Air_channel6',
	'Presc_sp_Flowrate_Air_channel7',
	'Presc_sp_Flowrate_Air_channel8',
	'Presc_sp_Flowrate_Air_channel9',
	'Presc_sp_Flowrate_Air_channel10',
	'Presc_sp_Flowrate_Air_channel11',
	'Presc_sp_Flowrate_Air_channel12']

	for missing_column in missing_columns:
		target_train[missing_column] = 0
		target_eval[missing_column] = 0
		target_test[missing_column] = 0
		if load_tar_lbls_loaders:
			target_train_lbls[missing_column] = 0
			target_eval_lbls[missing_column] = 0

	target_train[label_column] = target_train[label_column].astype(int)
	target_eval[label_column] = target_eval[label_column].astype(int)
	target_test[label_column] = target_test[label_column].astype(int)
	if load_tar_lbls_loaders:
		target_train_lbls[label_column] = target_train_lbls[label_column].astype(int)
		target_eval_lbls[label_column] = target_eval_lbls[label_column].astype(int)

	source_train["Domain"] = "Source"
	source_eval["Domain"] = "Source"
	source_test["Domain"] = "Source"

	target_train["Domain"] = "Target"
	target_eval["Domain"] = "Target"
	target_test["Domain"] = "Target"
	if load_tar_lbls_loaders:
		target_train_lbls['Domain'] = 'Target'
		target_eval_lbls['Domain'] = 'Target'

	meta_columns += ["Domain"]

	source_train_target_df = source_train[[label_column]]
	source_eval_target_df = source_eval[[label_column]]
	source_test_target_df = source_test[[label_column]]

	source_train_meta_df = source_train[meta_columns]
	source_eval_meta_df = source_eval[meta_columns]
	source_test_meta_df = source_test[meta_columns]

	target_train_target_df = target_train[[label_column]]
	target_eval_target_df = target_eval[[label_column]]
	target_test_target_df = target_test[[label_column]]

	target_train_meta_df = target_train[meta_columns]
	target_eval_meta_df = target_eval[meta_columns]
	target_test_meta_df = target_test[meta_columns]

	if load_tar_lbls_loaders:
		target_train_lbls_target_sr = target_train_lbls[[label_column]]
		target_train_lbls_meta_sr = target_train_lbls[meta_columns]
		target_eval_lbls_target_sr = target_eval_lbls[[label_column]]
		target_eval_lbls_meta_sr = target_eval_lbls[meta_columns]

	source_numeric_cols = [col for col in source_train.columns if col not in meta_columns+[label_column]]
	target_numeric_cols = [col for col in target_train.columns if col not in meta_columns+[label_column]]

	if prnt_processing_info:
		print("Before common cols - Source",source_train.shape)
		print("Before Common Cols - Target",target_train.shape)
	if load_common_cols:
		if prnt_processing_info:
			print("Loading common numeric cols")
			print("\tLenth of source numeric cols - ", len(source_numeric_cols))
			print("\tLenth of target numeric cols - ", len(target_numeric_cols))

		common_numeric_cols = utilis.Intersection(source_numeric_cols, target_numeric_cols)
		if prnt_processing_info:
			print("\tLenth of common numeric cols - ", len(common_numeric_cols))
		source_numeric_cols = common_numeric_cols
		target_numeric_cols = common_numeric_cols

		if prnt_processing_info:
			print("\tLenth of source numeric cols - ", len(source_numeric_cols))
			print("\tLenth of target numeric cols - ", len(target_numeric_cols))

	source_train = source_train[source_numeric_cols]
	source_eval = source_eval[source_numeric_cols]
	source_test = source_test[source_numeric_cols]

	target_train = target_train[target_numeric_cols]
	target_eval = target_eval[target_numeric_cols]
	target_test = target_test[target_numeric_cols]
	if load_tar_lbls_loaders:
		target_train_lbls = target_train_lbls[target_numeric_cols]
		target_eval_lbls = target_eval_lbls[target_numeric_cols]

	if prnt_processing_info:
		print("After common cols - Source",source_train.shape)
		print("After Common Cols - Target",target_train.shape)

	if prnt_processing_info:
		print("Before removing zero std - Source",source_train.shape)
		print("Before removing zero std - Target",target_train.shape)

	if remove_zero_std_cols:
		source_non_zero_cols = utilis.get_zero_std_cols(source_train)
		source_train = source_train[source_non_zero_cols]
		source_eval = source_eval[source_non_zero_cols]
		source_test = source_test[source_non_zero_cols]

		if load_common_cols:
			if prnt_processing_info:
				print("Removing target non zero cols based on source")
			target_train = target_train[source_non_zero_cols]
			target_eval = target_eval[source_non_zero_cols]
			target_test = target_test[source_non_zero_cols]
			if load_tar_lbls_loaders:
				target_train_lbls = target_train_lbls[source_non_zero_cols]
				target_eval_lbls = target_eval_lbls[source_non_zero_cols]
		else:
			if prnt_processing_info:
				print("Removing target non zero cols independently")
			target_non_zero_cols = utilis.get_zero_std_cols(target_train)
			target_train = target_train[target_non_zero_cols]
			target_eval = target_eval[target_non_zero_cols]
			target_test = target_test[target_non_zero_cols]
			if load_tar_lbls_loaders:
				target_train_lbls = target_train_lbls[target_non_zero_cols]
				target_eval_lbls = target_eval_lbls[target_non_zero_cols]

	if prnt_processing_info:
		print("After removing zero std - Source",source_train.shape)
		print("After removing zero std - Target",target_train.shape)

	configs['Source_no_of_features'] = int(source_train.shape[-1])
	configs['Target_no_of_features'] = int(target_train.shape[-1])
	print("No. of Features (Source Data) = "+str(configs['Source_no_of_features']))
	print("")
	print("No. of Features (Target Data) = "+str(configs['Target_no_of_features']))
	print("")

	source_scaler = utilis.get_scaler(source_train, scaling=configs['data_norm_mode'])
	source_scaler, source_train, source_eval, source_test = transform_data(source_scaler,source_train, source_eval, source_test)

	if configs['impose_scalers']:
		source_scaler, target_train, target_eval, target_test = transform_data(source_scaler,target_train, target_eval, target_test)
		if load_tar_lbls_loaders:
			target_train_lbls = pd.DataFrame(source_scaler.transform(target_train_lbls), columns=target_train_lbls.columns)
			target_eval_lbls = pd.DataFrame(source_scaler.transform(target_eval_lbls), columns=target_eval_lbls.columns)
	else:
		target_scaler = utilis.get_scaler(target_train, scaling=configs['data_norm_mode'])
		target_scaler, target_train, target_eval, target_test = transform_data(target_scaler,target_train, target_eval, target_test)
		if load_tar_lbls_loaders:
			target_train_lbls = pd.DataFrame(target_scaler.transform(target_train_lbls), columns=target_train_lbls.columns)
			target_eval_lbls = pd.DataFrame(target_scaler.transform(target_eval_lbls), columns=target_eval_lbls.columns)

	if prnt_processing_info:
		print("After Scaling - Source",source_train.shape)
		print("After Scaling - Target",target_train.shape)

	if configs['upsample']!=None and encoder_sr==None:
		if configs['seed_upsample']:
			rand_seed=configs['rand_seed']
		else:
			rand_seed=42
		sampler = utilis.get_sampler(sampler=configs['upsample'],rand_seed=rand_seed)
		source_train, source_train_target_df = sampler.fit_resample(source_train, source_train_target_df)
		source_train_meta_df = None

	if prnt_processing_info:
		print("After Resampling - Source",source_train.shape)
		print("After Resampling - Target",target_train.shape)

	if encoder_sr is not None:
		Encoder_src = encoder_sr[0]
		Encoder_tar = encoder_sr[1]

		feat_cols = [ 'feature'+str(i) for i in range(configs['latent_dim']) ]
		source_train = data_df_to_embeds_df_sr(Encoder=Encoder_src,df_sr=source_train,feat_cols=feat_cols)
		source_eval = data_df_to_embeds_df_sr(Encoder=Encoder_src,df_sr=source_eval,feat_cols=feat_cols)
		source_test = data_df_to_embeds_df_sr(Encoder=Encoder_src,df_sr=source_test,feat_cols=feat_cols)
		target_train = data_df_to_embeds_df_sr(Encoder=Encoder_tar,df_sr=target_train,feat_cols=feat_cols)
		target_eval = data_df_to_embeds_df_sr(Encoder=Encoder_tar,df_sr=target_eval,feat_cols=feat_cols)
		target_test = data_df_to_embeds_df_sr(Encoder=Encoder_tar,df_sr=target_test,feat_cols=feat_cols)
		if load_tar_lbls_loaders:
			target_train_lbls = data_df_to_embeds_df_sr(Encoder=Encoder_tar,df_sr=target_train_lbls,feat_cols=feat_cols)
			target_eval_lbls = data_df_to_embeds_df_sr(Encoder=Encoder_tar,df_sr=target_eval_lbls,feat_cols=feat_cols)

	want_ps_lbls=False
	if ps_lbls_dict!=None:
		want_ps_lbls=ps_lbls_dict['want_ps_lbls']

	psl_dict=None
	if want_ps_lbls:
		psl_dict={}
		psl_dict['wn_sr']=ps_lbls_dict['window_num']
		psl_dict['Source']={}
		psl_dict['Target']={}
		if ps_lbls_dict['window_set'] == 'Source':
			psl_dict['Source']['Train_df_sr'],psl_dict['Source']['Train_lbls'] = make_psuedo_labels_df_sr(df_sr=source_train,psl_dict=ps_lbls_dict)
			psl_dict['Source']['Eval_df_sr'],psl_dict['Source']['Eval_lbls'] = make_psuedo_labels_df_sr(df_sr=source_eval,psl_dict=ps_lbls_dict)
			psl_dict['Source']['Test_df_sr'],psl_dict['Source']['Test_lbls'] = make_psuedo_labels_df_sr(df_sr=source_test,psl_dict=ps_lbls_dict)
		else:
			psl_dict['Source']=None
		if ps_lbls_dict['window_set'] == 'Target':
			psl_dict['Target']['Train_df_sr'],psl_dict['Target']['Train_lbls'] = make_psuedo_labels_df_sr(df_sr=target_train,psl_dict=ps_lbls_dict)
			psl_dict['Target']['Eval_df_sr'],psl_dict['Target']['Eval_lbls'] = make_psuedo_labels_df_sr(df_sr=target_eval,psl_dict=ps_lbls_dict)
			psl_dict['Target']['Test_df_sr'],psl_dict['Target']['Test_lbls'] = make_psuedo_labels_df_sr(df_sr=target_test,psl_dict=ps_lbls_dict)
		else:
			psl_dict['Target']=None
	if psl_dict==None:
		psl_dict={}
		psl_dict['Source']=None
		psl_dict['Target']=None

	configs["Source_Train_Loader"] = get_data_loader(data_df=source_train, target_df=source_train_target_df, meta_df=source_train_meta_df, batch_size=configs['batch_size'], shuffle=True, psl_dict=psl_dict['Source'], keywrd='Train')
	configs = add_pos_neg_counts_sr(tar_df_sr=source_train_target_df,keywrd='Source_Train',configs=configs,psuedo_dict=psl_dict)
	configs["Source_Eval_Loader"] = get_data_loader(data_df=source_eval, target_df=source_eval_target_df, meta_df=source_eval_meta_df, batch_size=configs['batch_size'], psl_dict=psl_dict['Source'], keywrd='Eval')
	configs = add_pos_neg_counts_sr(tar_df_sr=source_eval_target_df,keywrd='Source_Eval',configs=configs,psuedo_dict=psl_dict)
	configs["Source_Test_Loader"] = get_data_loader(data_df=source_test, target_df=source_test_target_df, meta_df=source_test_meta_df, batch_size=configs['batch_size'], psl_dict=psl_dict['Source'], keywrd='Test')
	configs = add_pos_neg_counts_sr(tar_df_sr=source_test_target_df,keywrd='Source_Test',configs=configs,psuedo_dict=psl_dict)

	configs["Target_Train_Loader"]  = get_data_loader(data_df=target_train, target_df=target_train_target_df, meta_df=target_train_meta_df, batch_size=configs['batch_size'], shuffle=True, psl_dict=psl_dict['Target'], keywrd='Train')
	configs = add_pos_neg_counts_sr(tar_df_sr=target_train_target_df,keywrd='Target_Train',configs=configs,psuedo_dict=psl_dict)
	configs["Target_Eval_Loader"] = get_data_loader(data_df=target_eval, target_df=target_eval_target_df, meta_df=target_eval_meta_df, batch_size=configs['batch_size'], psl_dict=psl_dict['Target'], keywrd='Eval')
	configs = add_pos_neg_counts_sr(tar_df_sr=target_eval_target_df,keywrd='Target_Eval',configs=configs,psuedo_dict=psl_dict)
	configs["Target_Test_Loader"] = get_data_loader(data_df=target_test, target_df=target_test_target_df, meta_df=target_test_meta_df, batch_size=configs['batch_size'], psl_dict=psl_dict['Target'], keywrd='Test')
	configs = add_pos_neg_counts_sr(tar_df_sr=target_test_target_df,keywrd='Target_Test',configs=configs,psuedo_dict=psl_dict)

	if load_tar_lbls_loaders:
		configs['Target_Train_lbls_Loader'] = get_data_loader(data_df=target_train_lbls, target_df=target_train_lbls_target_sr, meta_df=target_train_lbls_meta_sr, batch_size=configs['batch_size'], shuffle=True, psl_dict=None)
		configs = add_pos_neg_counts_sr(tar_df_sr=target_train_lbls_target_sr,keywrd='Target_Train_labels',configs=configs,psuedo_dict=None)
		configs['Target_Test_lbls_Loader'] = get_data_loader(data_df=target_eval_lbls, target_df=target_eval_lbls_target_sr, meta_df=target_eval_lbls_meta_sr, batch_size=configs['batch_size'], psl_dict=None)
		configs = add_pos_neg_counts_sr(tar_df_sr=target_eval_lbls_target_sr,keywrd='Target_Test_labels',configs=configs,psuedo_dict=None)

	if (encoder_sr is not None) and (configs['to_save_embeds'] or configs['to_plt_tsne']!='None'):
		configs['Source_configs'] = json.load(open(configs['Source_configs_path']))
		configs['Target_configs'] = json.load(open(configs['Target_configs_path']))

		print("Making Train Reps Dataframe (Source) --> ")
		configs['Source_train_df_tsne'] = make_df_for_hcf_tsne_sr(configs=configs,domain='Source',data=pd.read_feather(configs['Source_train_data_path']),meta_df_sr=source_train_meta_df,embeds_df_sr=source_train,lbls_df_sr=source_train_target_df)
		print("")
		print("Making Eval Reps Dataframe (Source) --> ")
		configs['Source_eval_df_tsne'] = make_df_for_hcf_tsne_sr(configs=configs,domain='Source',data=pd.read_feather(configs['Source_eval_data_path']),meta_df_sr=source_eval_meta_df,embeds_df_sr=source_eval,lbls_df_sr=source_eval_target_df)
		print("")
		print("Making Test Reps Dataframe (Source) --> ")
		configs['Source_test_df_tsne'] = make_df_for_hcf_tsne_sr(configs=configs,domain='Source',data=pd.read_feather(configs['Source_test_data_path']),meta_df_sr=source_test_meta_df,embeds_df_sr=source_test,lbls_df_sr=source_test_target_df)
		print("")
		print("Making Train Reps Dataframe (Target) --> ")
		configs['Target_train_df_tsne'] = make_df_for_hcf_tsne_sr(configs=configs,domain='Target',data=pd.read_feather(configs['Target_train_data_path']),meta_df_sr=target_train_meta_df,embeds_df_sr=target_train,lbls_df_sr=target_train_target_df)
		print("")
		print("Making Eval Reps Dataframe (Target) --> ")
		configs['Target_eval_df_tsne'] = make_df_for_hcf_tsne_sr(configs=configs,domain='Target',data=pd.read_feather(configs['Target_eval_data_path']),meta_df_sr=target_eval_meta_df,embeds_df_sr=target_eval,lbls_df_sr=target_eval_target_df)
		print("")
		print("Making Test Reps Dataframe (Target) --> ")
		configs['Target_test_df_tsne'] = make_df_for_hcf_tsne_sr(configs=configs,domain='Target',data=pd.read_feather(configs['Target_test_data_path']),meta_df_sr=target_test_meta_df,embeds_df_sr=target_test,lbls_df_sr=target_test_target_df)
		
	return configs

# %%
def get_hcf_data_to_loaders(configs):
	configs,psl_dict_sr = make_ps_lbls_dict_sr(configs=configs)
	configs = get_hcf_dataloaders(configs=configs,ps_lbls_dict=psl_dict_sr,prnt_processing_info=configs['prnt_processing_info'])
	return configs

# %%
def get_hcf_data_to_embed_loaders(configs,Encoder):

	configs = get_hcf_dataloaders(configs=configs,encoder_sr=Encoder,prnt_processing_info=configs['prnt_processing_info'])
	if configs['to_save_embeds'] or configs['to_plt_tsne']!='None':
		utilis.append_colnms_sr(data=configs['Source_train_df_tsne'],nm='src')
		utilis.append_colnms_sr(data=configs['Source_eval_df_tsne'],nm='src')
		utilis.append_colnms_sr(data=configs['Source_test_df_tsne'],nm='src')
		utilis.append_colnms_sr(data=configs['Target_train_df_tsne'],nm='tar')
		utilis.append_colnms_sr(data=configs['Target_eval_df_tsne'],nm='tar')
		utilis.append_colnms_sr(data=configs['Target_test_df_tsne'],nm='tar')

		if configs['Target_train_df_tsne'] is None:
			configs['Train_tsne_df'] = configs['Source_train_df_tsne']
			configs['Test_tsne_df'] = configs['Source_eval_df_tsne']
			configs['Eval_tsne_df'] = configs['Source_test_df_tsne']
		else:
			configs['Train_tsne_df'] = pd.concat([configs['Source_train_df_tsne'],configs['Target_train_df_tsne']],axis=1,ignore_index=False)
			configs['Eval_tsne_df'] = pd.concat([configs['Source_eval_df_tsne'],configs['Target_eval_df_tsne']],axis=1,ignore_index=False)
			configs['Test_tsne_df'] = pd.concat([configs['Source_test_df_tsne'],configs['Target_test_df_tsne']],axis=1,ignore_index=False)

		if configs['to_save_embeds']:
			configs['Train_tsne_df'].to_csv(configs['save_train_embeds_path'],index=False)
			configs['Eval_tsne_df'].to_csv(configs['save_eval_embeds_path'],index=False)
			configs['Test_tsne_df'].to_csv(configs['save_test_embeds_path'],index=False)

		if configs['to_plt_tsne']!='None':
			configs['tsne_df'] = pd.concat([configs['Train_tsne_df'],configs['Eval_tsne_df'],configs['Test_tsne_df']],axis=0,ignore_index=True)
			print("Net df shape (for t-SNE Plots) = "+str(configs['tsne_df'].shape))

	return configs
