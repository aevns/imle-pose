from math import log
from math import floor
import torch
import torch.nn as nn

def gaussian_entropy(pred):
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]

    return torch.sum(torch.log(torch.det(cov_mat))/2 + 2.8378770664093455, dim=(-1))

def gaussian_nll(pred, x):
    pose = pred['pose'][:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]
    
    dif = torch.reshape(x['pose'] - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
    q = q.view(q.shape[0], q.shape[1])
    return torch.sum((torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455, dim=(-1))

def heatmap_target_mse(pred, x):
    return 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred['heatmap'], x['target']), dim=(-3,-2,-1))

def heatmap_target_dkl(pred, x):
    target = x['target'] / torch.sum(x['target'], dim=(-3,-2,-1)).view(-1,1,1,1)
    return torch.sum(target * torch.log(pred['heatmap'] / target), dim=(-3,-2,-1))