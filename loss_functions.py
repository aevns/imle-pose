import torch
import torch.nn as nn

def expected_gaussian_entropy(pred):
    pose = pred['pose'][:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]

    return torch.log(torch.det(cov_mat))/2 + 1 + 1.8378770664093455

def gaussian_entropy(pred, x):
    pose = pred['pose'][:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred['pose'][:,:,cov_idx]
    
    dif = torch.reshape(x['pose'] - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    dif_t = torch.reshape(x['pose'] - pose, (pose.shape[0], pose.shape[1], 1, pose.shape[2]))
    q = torch.matmul(dif_t, torch.matmul(torch.inverse(cov_mat), dif))
    q = torch.reshape(q, (q.shape[0], q.shape[1]))
    return torch.mean(q + torch.log(torch.det(cov_mat)), dim=(-1))/2 + 1.8378770664093455

def heatmap_target_mse(pred, x):
    return 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred['heatmap'], x['target']), dim=(-3,-2,-1))