import torch.nn.functional as F
from utils.metrics import topk_corrects
import torch
from torch.autograd import grad
import numpy as np

from core.utils import gather_flat_grad,neumann_hyperstep_preconditioner
from core.utils import get_trainable_hyper_params,assign_hyper_gradient
from dataset.train_utils.utilis import *
from sklearn.metrics import roc_auc_score, confusion_matrix
import wandb

def make_one_hot(label):
    x = torch.zeros((label.size(0), 2))
    for i, l in enumerate(label):
        if l.item() == 0:
            x[i][1] = 1
        else:
            x[i][0] = 1
    return x



def train_epoch(
    cur_epoch, model, args,
    low_loader, low_criterion , low_optimizer, low_params=None,
    up_loader=None, up_optimizer=None, up_criterion=None, up_params=None, encoder=None
    ):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    ARCH_EPOCH=args["up_configs"]["start_epoch"]
    ARCH_END=args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL=args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL=args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE=args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE=args["up_configs"]["val_batches"]
    device=args["device"]
    is_up=(cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
        ((cur_epoch+1) % ARCH_EPOCH_INTERVAL) == 0
    
    if is_up:
        print('lower lr: ',low_optimizer.param_groups[0]['lr'],'  upper lr: ',up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        low_iter_alt=iter(low_loader)
    else:
        print('lr: ',low_optimizer.param_groups[0]['lr'])
    
    model.train()
    # encoder.train()
    total_correct=0.
    total_sample=0.
    total_loss=0.
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 2*((num_classes-1)//group_size)+1
    use_reg=True

    d_train_loss_d_w = torch.zeros(num_weights,device=device)

    for cur_iter, (low_data, low_targets, _, _) in enumerate(low_loader):
        #print(cur_iter)
        # Transfer the data to the current GPU device
        # if cur_iter%5==0:
        #     print(cur_iter,len(low_loader))
        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device, non_blocking=True)
        # print('Before encoder', low_data.shape)
        low_data = encoder(low_data) if encoder is not None else low_data
        # print('After encoder', low_data.shape)
        if args['model_type'] == 'tsc':
            low_data = low_data.view(-1, 19, 301)

        low_targets = low_targets.view(-1)

        # Update architecture
        if is_up:
            model.train()
            up_optimizer.zero_grad()
            if cur_iter%ARCH_INTERVAL==0:
                for _ in range(ARCH_TRAIN_SAMPLE):
                    try:
                        low_data_alt, low_targets_alt, _, _ = next(low_iter_alt)
                    except StopIteration:
                        low_iter_alt = iter(low_loader)
                        low_data_alt, low_targets_alt, _, _ = next(low_iter_alt)
                    low_data_alt, low_targets_alt = low_data_alt.to(device=device, non_blocking=True), low_targets_alt.to(device=device, non_blocking=True)
                    low_data_alt = encoder(low_data_alt) if encoder is not None else low_data_alt
                    low_targets_alt = low_targets_alt.view(-1)

                    if args['model_type'] == 'tsc':
                        low_data_alt = low_data_alt.view(-1, 19, 301)

                    low_optimizer.zero_grad()

                    if args['model_type'] == 'encoder':
                        with torch.backends.cudnn.flags(enabled=False):
                            low_preds=model(low_data_alt)
                    else:
                        low_preds = model(low_data_alt)


                    low_loss=low_criterion(low_preds,low_targets_alt,low_params,group_size=group_size) 
                    d_train_loss_d_w+=gather_flat_grad(grad(low_loss,model.parameters(),create_graph=True))
                    #print(cur_iter_alt)
                d_train_loss_d_w /= ARCH_TRAIN_SAMPLE
                d_val_loss_d_theta = torch.zeros(num_weights,device=device)
                # d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()

                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        up_data, up_targets, _, _ = next(up_iter)
                    except StopIteration:
                        up_iter = iter(up_loader)
                        up_data, up_targets, _, _ = next(up_iter)
                #for _,(up_data,up_targets) in enumerate(up_loader):
                    up_data, up_targets = up_data.to(device=device, non_blocking=True), up_targets.to(device=device, non_blocking=True)
                    up_data = encoder(up_data) if encoder is not None else up_data
                    up_targets = up_targets.view(-1)

                    if args['model_type'] == 'tsc':
                        up_data = up_data.view(-1, 19, 301)

                    model.zero_grad()
                    low_optimizer.zero_grad()


                    if args['model_type'] == 'encoder':
                        with torch.backends.cudnn.flags(enabled=False):
                            up_preds = model(up_data)
                    else:
                        up_preds = model(up_data)

                    up_loss = up_criterion(up_preds,up_targets,up_params,group_size=group_size)
                    d_val_loss_d_theta += gather_flat_grad(grad(up_loss, model.parameters(), retain_graph=use_reg))
                    # if use_reg:
                    #     direct_grad+=gather_flat_grad(grad(up_loss, get_trainable_hyper_params(up_params), allow_unused=True))
                    #     direct_grad[direct_grad != direct_grad] = 0
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                #direct_grad/=ARCH_VAL_SAMPLE
                preconditioner = d_val_loss_d_theta
                
                preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 1.0,
                                                                5, model)
                indirect_grad = gather_flat_grad(
                    grad(d_train_loss_d_w, get_trainable_hyper_params(up_params), grad_outputs=preconditioner.view(-1),allow_unused=True))
                hyper_grad=-indirect_grad#+direct_grad
                up_optimizer.zero_grad()
                assign_hyper_gradient(up_params,hyper_grad,num_classes)
                up_optimizer.step()
                d_train_loss_d_w = torch.zeros(num_weights,device=device)
        
        if is_up:
            try:
                up_data, up_targets, _, _ = next(up_iter)
            except StopIteration:
                up_iter = iter(up_loader)
                up_data, up_targets, _, _ = next(up_iter)
            up_data, up_targets = up_data.to(device=device, non_blocking=True), up_targets.to(device=device, non_blocking=True)
            up_data = encoder(up_data) if encoder is not None else up_data
            up_targets = up_targets.view(-1)
            if args['model_type'] == 'tsc':
                up_data = up_data.view(-1, 19, 301)


            if args['model_type'] == 'encoder':
                with torch.backends.cudnn.flags(enabled=False):
                    up_preds=model(up_data)
            else:
                up_preds = model(up_data)

            up_loss=up_criterion(up_preds,up_targets,up_params,group_size=group_size)
            up_optimizer.zero_grad()
            up_loss.backward()
            up_optimizer.step()


        # Perform the forward pass
        low_preds = model(low_data)

        # Compute the loss
        # print('target', low_targets.shape)
        # print('prediction', low_preds.shape)

        loss = low_criterion(low_preds, low_targets, low_params,group_size=group_size)

        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1] 
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct+=top1_correct
        total_sample+=mb_size
        total_loss+=loss*mb_size
    # Log epoch stats
    wandb.log({'Loss in lower optimization level': total_loss / total_sample})
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')



# @torch.no_grad()
# def eval_epoch(data_loader, model, criterion, cur_epoch, text, args, params=None):
#     num_classes=args["num_classes"]
#     group_size=args["group_size"]
#     model.eval()
#     correct=0.
#     total=0.
#     loss=0.
#     class_correct=np.zeros(num_classes,dtype=float)
#     class_total=np.zeros(num_classes,dtype=float)
#     confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
#     for cur_iter, (data, targets) in enumerate(data_loader):
#         if cur_iter%5==0:
#             print(cur_iter,len(data_loader))
#         data, targets = data.cuda(), targets.cuda(non_blocking=True)
#         logits = model(data)
         
#         preds = logits.data.max(1)[1]
#         mb_size = data.size(0)
#         loss+=criterion(logits,targets,params,group_size=group_size).item()*mb_size

#         total+=mb_size
#         correct+=preds.eq(targets.data.view_as(preds)).sum().item()


        
#         for i in range(num_classes):
#             indexes=np.where(targets.cpu().numpy()==i)[0]
#             class_total[i]+=indexes.size
#             class_correct[i]+=preds[indexes].eq(targets[indexes].data.view_as(preds[indexes])).sum().item()

#     text=f'{text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.} Balanced ACC = {np.mean(class_correct/class_total*100.)} \n Class wise = {class_correct/class_total*100.}'
#     print(text)
#     return text, loss/total, 100.-correct/total*100., 100.-float(np.mean(class_correct/class_total*100.))

@torch.no_grad()
def eval_epoch(data_loader, model, criterion, cur_epoch, text, args, params=None, encoder=None):
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    model.eval()
    correct=0.
    total=0.
    loss=0.
    class_correct=np.zeros(num_classes,dtype=float)
    class_total=np.zeros(num_classes,dtype=float)
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    for cur_iter, (data, targets, _, _) in enumerate(data_loader):
        # if cur_iter%5==0:
        #     print(cur_iter,len(data_loader))
        data, targets = data.cuda(), targets.cuda(non_blocking=True)
        data = encoder(data) if encoder is not None else data
        targets = targets.view(-1)

        if args['model_type'] == 'tsc':
            data = data.view(-1, 19, 301)

        logits = model(data)
         
        preds = logits.data.max(1)[1]
        for t, p in zip(targets.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        mb_size = data.size(0)
        loss+=criterion(logits,targets,params,group_size=group_size).item()*mb_size
    class_correct=confusion_matrix.diag().cpu().numpy()
    class_total=confusion_matrix.sum(1).cpu().numpy()
    total=confusion_matrix.sum().cpu().numpy()
    correct=class_correct.sum()

    text=f'{text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.} Balanced ACC = {np.mean(class_correct/class_total*100.)} \n Class wise = {class_correct/class_total*100.}'
    print(text)
    return text, loss/total, 100.-correct/total*100., 100.-float(np.mean(class_correct/class_total*100.)), (100.-class_correct/class_total*100.).tolist()


def evaluate_model_tvarit(data_loader, model, criterion, cur_epoch, text, args, encoder=None):
    num_classes = args["num_classes"]
    group_size = args["group_size"]
    model.eval()
    correct = 0.
    total = 0.
    loss = 0.
    class_correct = np.zeros(num_classes, dtype=float)
    class_total = np.zeros(num_classes, dtype=float)
    #confusion_matrix = torch.zeros(num_classes, num_classes).cuda()

    target_list = []
    pred_list = []

    for cur_iter, (data, targets, _, _) in enumerate(data_loader):
        # if cur_iter%5==0:
        #     print(cur_iter,len(data_loader))
        data, targets = data.cuda(), targets.cuda(non_blocking=True)
        data = encoder(data) if encoder is not None else data
        targets = targets.view(-1)

        if args['model_type'] == 'tsc':
            data = data.view(-1, 19, 301)

        logits = model(data)

        preds = torch.softmax(logits, dim=1)[:,1]

        target_temp = targets.cpu().detach().numpy().tolist()
        preds_temp = preds.cpu().detach().numpy().tolist()

        target_list.extend(target_temp)
        pred_list.extend(preds_temp)
    
    target_arr = np.array(target_list)
    pred_arr = np.array(pred_list)
    auc_val = roc_auc_score(target_arr, pred_arr)
    if True:
        best_thresh = plot_roc_curve(target_arr, pred_arr, text, None)
        best_thresh = 0.2
        pred_arr[pred_arr > best_thresh] = 1
        pred_arr[pred_arr < best_thresh] = 0
        conf_matrix = confusion_matrix(target_arr, pred_arr)
        print('Confusion Matrix for {}'.format(text), conf_matrix)
    print('AUROC Value for {}'.format(text), auc_val)
    return auc_val

def evaluate_old_model_tvarit(data_loader, model, criterion, cur_epoch, text, args, encoder=None):
    num_classes = args["num_classes"]
    group_size = args["group_size"]
    model.eval()
    correct = 0.
    total = 0.
    loss = 0.
    class_correct = np.zeros(num_classes, dtype=float)
    class_total = np.zeros(num_classes, dtype=float)
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()

    target_list = []
    pred_list = []

    for cur_iter, (data, targets, _, _) in enumerate(data_loader):
        # if cur_iter%5==0:
        #     print(cur_iter,len(data_loader))
        data, targets = data.cuda(), targets.cuda(non_blocking=True)
        data = encoder(data) if encoder is not None else data
        if args['model_type'] == 'tsc':
            data = data.view(-1, 19, 301)

        targets = targets.view(-1)
        preds = model(data).view(-1)

        target_temp = targets.cpu().detach().numpy().tolist()
        preds_temp = preds.cpu().detach().numpy().tolist()

        target_list.extend(target_temp)
        pred_list.extend(preds_temp)

    auc_val = roc_auc_score(np.array(target_list), np.array(pred_list))
    print('AUROC Value for {}'.format(text), auc_val)

    return auc_val
