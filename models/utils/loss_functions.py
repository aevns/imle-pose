from math import log
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_entropy(pred):
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]
    if pred['pose'].shape[2] == 6:
        label_preds = pred['pose'][:,:,5]
    label_entropy = -label_preds * torch.log(label_preds) - (1 - label_preds) * torch.log(1 - label_preds)
    asdf = torch.log(torch.det(cov_mat))/2
    entropy = label_preds * (torch.log(torch.det(cov_mat))/2 + 2.8378770664093455 - torch.log(label_preds)) + (1 - label_preds) * (- torch.log(1 - label_preds))
    return torch.sum(entropy, dim=(-1))

def label_loss(pred, x):
    labeled = pred['pose'][:,:,5]
    return torch.sum((labeled - 0.5)**2, dim=(-1))

def gaussian_nll(pred, x):
    
    pose = pred['pose'][:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]
    
    gt_pose = x['pose'][:,:,0:2]

    dif = torch.reshape(gt_pose - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
    q = q.view(q.shape[0], q.shape[1])
    return torch.sum((torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455, dim=(-1))

def heatmap_target_mse(pred, x):
    return torch.mean(nn.MSELoss(reduction='none')(pred['heatmap'], x['target']), dim=(-3,-2,-1))

def heatmap_target_dkl(pred, x):
    eps = torch.tensor([1E-8], device=x['target'].device)
    target = torch.max(x['target'] / torch.sum(x['target'], dim=(-3,-2,-1)).view(-1,1,1,1), eps)
    return torch.sum(target * torch.log(torch.max(pred['heatmap'], eps) / target), dim=(-3,-2,-1))

def heatmap_target_prob_prod(pred, x):
    eps = torch.tensor([1E-8], device=x['target'].device)
    target = torch.max(x['target'] / torch.sum(x['target'], dim=(-3,-2,-1)).view(-1,1,1,1), eps)
    return torch.prod(torch.sum(target * torch.max(pred['heatmap'], eps), dim=(-2,-1)), dim=-1)