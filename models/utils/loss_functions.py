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
    labeled = pred['pose'][:,:,5]
    
    gt_pose = x['pose'][:,:,0:2]
    if x['pose'].shape[-1] == 3:
        mask = (x['pose'][:,:,2] != 0)

        label_loss = -torch.sum(torch.log(labeled[mask])) - torch.sum(torch.log((1 - labeled)[~mask]))
        # for use with imprefect ground truth masking
        #label_loss = -mask * torch.log(labeled / mask) - (1 - mask) * torch.log((1 - labeled) / (1 - mask))
    else:
        mask = 1
        label_loss = 0

    dif = torch.reshape(gt_pose - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
    q = q.view(q.shape[0], q.shape[1])
    return torch.sum(mask * (torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455, dim=(-1)) + label_loss

def heatmap_target_mse(pred, x):
    return 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred['heatmap'], x['target']), dim=(-3,-2,-1))

def heatmap_target_dkl(pred, x):
    eps = torch.tensor([1E-8], device=x['target'].device)
    target = torch.max(x['target'] / torch.sum(x['target'], dim=(-3,-2,-1)).view(-1,1,1,1), eps)
    return torch.sum(target * torch.log(torch.max(pred['heatmap'], eps) / target), dim=(-3,-2,-1))

def heatmap_target_prob_prod(pred, x):
    eps = torch.tensor([1E-8], device=x['target'].device)
    target = torch.max(x['target'] / torch.sum(x['target'], dim=(-3,-2,-1)).view(-1,1,1,1), eps)
    return torch.prod(torch.sum(target * torch.max(pred['heatmap'], eps), dim=(-2,-1)), dim=-1)