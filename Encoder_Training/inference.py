# %%
# *********************** Importing Essential Libraries *************************
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

import dataset.train_utils.test_procedures as test_procedures
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
def confusion_matrix_frm_tpr_fpr(tpr,fpr,op,on): # op = observed positives; on = observed negatives
	cfm = np.zeros([2,2])
	tp = int(tpr*op/100)
	tn = int((100-fpr)*on/100)
	fp = on - tn
	fn = op - tp
	cfm[0,0] = tn
	cfm[0,1] = fp
	cfm[1,0] = fn
	cfm[1,1] = tp
	return cfm

# %%
def print_metrics(l,spec,sens,prec,f1,auc,nm,op,on,lam):
	print("Results on "+nm+" Dataset")
	print("Mean Loss --> "+str(l))
	print("AUC Score --> "+str(auc))
	print("Best Threshold(s) --> "+str(lam))
	if not ((sens is None) or (spec is None) or (op is None) or (on is None)):
		print(confusion_matrix_frm_tpr_fpr(tpr=sens,fpr=100-spec,op=op,on=on))
	print("Specificity (in %) = "+str(spec))
	print("Sensitivity (aka Recall) (in %) = "+str(sens))
	print("Precision (in %) = "+str(prec))
	print("F1 Score (in %) = "+str(f1))

# %%
def infer_classifier(Classifier,configs):
	thresh_tar = configs['best_threshold']
	fwd_tar=False
	if "best_threshold_tar" in configs.keys():
		thresh_tar=configs['best_threshold_tar']
		fwd_tar=True

	probab_plts_dict = None
	plt_roc_dict={}
	plt_roc_dict['path'] = configs['save_plot_path']
	plt_roc_dict['nm'] = 'Source_Train'
	if configs['want_probab_plots']:
		probab_plts_dict = {}
		probab_plts_dict['path'] = configs['save_probab_plots_path']
		probab_plts_dict['nm'] = plt_roc_dict['nm']
	_,l,spec,sens,prec,f1,auc = test_procedures.test_classification_scores_plots(Model=Classifier,loader=configs['Source_Train_Loader'],configs=configs,threshold=configs['best_threshold'],plt_roc_dict=plt_roc_dict,probab_plts_dict = probab_plts_dict)
	print_metrics(l=l,spec=spec,sens=sens,prec=prec,f1=f1,auc=auc,nm='Source_Train',op=configs['Source_Train_Positives'],on=configs['Source_Train_Negatives'],lam=configs['best_threshold'])
	print("")
	plt_roc_dict['nm'] = 'Source_Eval'
	if configs['want_probab_plots']:
		probab_plts_dict['nm'] = plt_roc_dict['nm']
	_,l,spec,sens,prec,f1,auc = test_procedures.test_classification_scores_plots(Model=Classifier,loader=configs['Source_Eval_Loader'],configs=configs,threshold=configs['best_threshold'],plt_roc_dict=plt_roc_dict,probab_plts_dict = probab_plts_dict)
	print_metrics(l=l,spec=spec,sens=sens,prec=prec,f1=f1,auc=auc,nm='Source_Eval',op=configs['Source_Eval_Positives'],on=configs['Source_Eval_Negatives'],lam=configs['best_threshold'])
	print("")
	plt_roc_dict['nm'] = 'Source_Test'
	if configs['want_probab_plots']:
		probab_plts_dict['nm'] = plt_roc_dict['nm']
	_,l,spec,sens,prec,f1,auc = test_procedures.test_classification_scores_plots(Model=Classifier,loader=configs['Source_Test_Loader'],configs=configs,threshold=configs['best_threshold'],plt_roc_dict=plt_roc_dict,probab_plts_dict = probab_plts_dict)
	print_metrics(l=l,spec=spec,sens=sens,prec=prec,f1=f1,auc=auc,nm='Source_Test',op=configs['Source_Test_Positives'],on=configs['Source_Test_Negatives'],lam=configs['best_threshold'])
	print("")
	plt_roc_dict['nm'] = 'Target_Train'
	if configs['want_probab_plots']:
		probab_plts_dict['nm'] = plt_roc_dict['nm']
	_,l,spec,sens,prec,f1,auc = test_procedures.test_classification_scores_plots(Model=Classifier,loader=configs['Target_Train_Loader'],configs=configs,threshold=thresh_tar,plt_roc_dict=plt_roc_dict,is_tar=fwd_tar,probab_plts_dict = probab_plts_dict)
	print_metrics(l=l,spec=spec,sens=sens,prec=prec,f1=f1,auc=auc,nm='Target_Train',op=configs['Target_Train_Positives'],on=configs['Target_Train_Negatives'],lam=thresh_tar)
	print("")
	plt_roc_dict['nm'] = 'Target_Eval'
	if configs['want_probab_plots']:
		probab_plts_dict['nm'] = plt_roc_dict['nm']
	_,l,spec,sens,prec,f1,auc = test_procedures.test_classification_scores_plots(Model=Classifier,loader=configs['Target_Eval_Loader'],configs=configs,threshold=thresh_tar,plt_roc_dict=plt_roc_dict,is_tar=fwd_tar,probab_plts_dict = probab_plts_dict)
	print_metrics(l=l,spec=spec,sens=sens,prec=prec,f1=f1,auc=auc,nm='Target_Eval',op=configs['Target_Eval_Positives'],on=configs['Target_Eval_Negatives'],lam=thresh_tar)
	print("")
	plt_roc_dict['nm'] = 'Target_Test'
	if configs['want_probab_plots']:
		probab_plts_dict['nm'] = plt_roc_dict['nm']
	_,l,spec,sens,prec,f1,auc = test_procedures.test_classification_scores_plots(Model=Classifier,loader=configs['Target_Test_Loader'],configs=configs,threshold=thresh_tar,plt_roc_dict=plt_roc_dict,is_tar=fwd_tar,probab_plts_dict = probab_plts_dict)
	print_metrics(l=l,spec=spec,sens=sens,prec=prec,f1=f1,auc=auc,nm='Target_Test',op=configs['Target_Test_Positives'],on=configs['Target_Test_Negatives'],lam=thresh_tar)
