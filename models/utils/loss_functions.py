import torch
import torch.nn as nn

def heatmap_target_mse(pred, x):
    return 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, x['target']), dim=(-3,-2,-1))

def gaussian_entropy(pred, x):
    pose = pred[:,:,0:2]
    cov_idx = torch.tensor([[2,4],[4,3]])
    cov_mat = pred[:,:,cov_idx]
    
    dif = torch.reshape(x['pose'] - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
    dif_t = torch.reshape(x['pose'] - pose, (pose.shape[0], pose.shape[1], 1, pose.shape[2]))
    q = torch.matmul(dif_t, torch.matmul(torch.inverse(cov_mat), dif))
    q = torch.reshape(q, (q.shape[0], q.shape[1]))
    return torch.mean(q + torch.log(torch.det(cov_mat)), dim=(-1))/2 + 1.8378770664093455

def heatmap_gaussian_fit(pred):
    n, c, h, w = pred.shape

    max_ = torch.max(torch.max(pred, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
    z = torch.sum(torch.exp(pred - max_), (2, 3)).view(n, c, 1, 1)
    h_norm = torch.exp(pred - max_) / z

    y_vals = torch.linspace(0, h-1, h, device=pred.device).unsqueeze(-1)
    x_vals = torch.linspace(0, w-1, w, device=pred.device).unsqueeze(-2)

    y_means = torch.sum(h_norm * y_vals, dim = (2, 3))
    x_means = torch.sum(h_norm * x_vals, dim = (2, 3))
    
    x_var = 1 + torch.sum(h_norm * (x_vals - x_means.view(n, c, 1, 1))**2, dim=(2,3))
    xy_covar = 0.8 * torch.sum(h_norm * (x_vals - x_means.view(n, c, 1, 1)) * (y_vals - y_means.view(n, c, 1, 1)), dim=(2,3))
    y_var = 1 + torch.sum(h_norm * (y_vals - y_means.view(n, c, 1, 1))**2, dim=(2,3))

    return torch.stack((x_means, y_means, x_var, y_var, xy_covar), -1)

def heatmap_gaussian_fit_entropy(pred, x):
    return gaussian_entropy(heatmap_gaussian_fit(pred), x)

def heatmaps_normalized(pred):
    n, c, h, w = pred.shape

    max_ = torch.max(torch.max(pred, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
    z = torch.sum(torch.exp(pred - max_), (2, 3)).view(n, c, 1, 1)
    h_norm = torch.exp(pred - max_) / z

    return h_norm
