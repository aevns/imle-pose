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

def gaussian_nll(pred, x):
    
    pose = pred['pose'][:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]
    labeled = pred['pose'][:,:,5]
    
    gt_pose = x['pose'][:,:,0:2]

    dif = torch.reshape(gt_pose - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
    q = q.view(q.shape[0], q.shape[1])
    return torch.sum((torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455, dim=(-1))


def gaussian_nll_label(pred, x):
    
    pose = pred['pose'][:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]
    labeled = pred['pose'][:,:,5]
    
    gt_pose = x['pose'][:,:,0:2]
    mask = x['pose'][:,:,5]
    
    eps = torch.tensor([1E-4], device=x['target'].device)
    labeled =  1 - 1 / (torch.sum(torch.exp(pred), dim=(2, 3)) + 1)
    label_loss = -torch.log(1 - labeled * (1 - eps))
    label_loss[mask] = -torch.log(eps + labeled[mask] * (1 - eps))

    #label_loss = -torch.sum(mask * torch.log(labeled / mask) + (1 - mask) * torch.log((1 - labeled) / (1 - mask)), dim=-1)

    dif = torch.reshape(gt_pose - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
    q = q.view(q.shape[0], q.shape[1])
    return torch.sum(mask * (torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455, dim=(-1)) + torch.sum(label_loss, dim=(-1))

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

def heatmap_log_target_dkl(pred, x):
    if pred['heatmap'].dim() == 4:
        B, K, H, W = pred['heatmap'].shape
        dist = torch.log_softmax(pred['heatmap'].view(B, K, H * W), dim=(-1)).view(B, K, H, W)
    elif pred['heatmap'].dim() == 3:
        K, H, W = pred['heatmap'].shape
        dist = torch.log_softmax(pred['heatmap'].view(K, H * W), dim=(-1)).view(K, H, W)
    target = x['log_target']
    loss = F.kl_div(dist, target, log_target=True, reduction='none').sum(dim=(-2, -1))
    loss = torch.sum(loss, dim=(-1))
    # label loss is used to ensure outputs do not get scaled arbitrarily
    # currently assumes all joints are guaranteed to be labeled (labeled should be 1)
    #labeled =  1 - 1 / (torch.sum(torch.exp(pred['heatmap']), dim=(-2, -1)) + 1)
    #label_loss = -torch.log(1E-4 + labeled * (1 - 1E-4))
    return torch.sum(loss)# + torch.sum(label_loss)