import json
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torchvision
import torchvision.transforms as transforms

from torch import dtype, uint8
from tqdm import tqdm
from collections import defaultdict

import models.utils.loss_functions as lf

from dataset import HDF5Sampler
from dataset import HDF5Dataset
from models.unet import UNet
from models.unet_pretrained import UNetPretrained

val_data = HDF5Dataset(
    "data/complete/compressed/val.hdf5",
    generate_heatmaps=False,
    device="cuda:0")

val_sampler = HDF5Sampler(
    data_source=val_data)
    
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = 1,
    num_workers=0,
    sampler=val_sampler)

network = UNet(lf.gaussian_nll, val_data.image_size, 8).cuda()
network.train(False)

state_dict = torch.load("output/label_0/state_dict/network_61.pth")
network.load_state_dict(state_dict)
network.training = False

with torch.no_grad():
    samples = 100
    val_iter = iter(val_loader)
    batch = next(val_iter)
    batch_repeat = {
        'image': batch['image'].expand(samples, -1, -1, -1),
        'pose': batch['pose'].expand(samples, -1, -1),
        'target': batch['target'].expand(samples)}
    predictions = network.get_sample(batch_repeat)

# plotting utility functions

cmap = plt.get_cmap('prism')
joint_names = val_data.keypoints
bone_indices = val_data.skeleton
joint_colors = torch.zeros((len(joint_names), 3))
bone_colors = torch.zeros((len(bone_indices), 3))
for i in range(0, len(joint_names)):
    joint_colors[i] = torch.tensor(cmap(i * cmap.N // len(joint_names))[:3])
for j in range(0, len(bone_colors)):
    bone_colors[j] = (joint_colors[bone_indices[j,0]-1] + joint_colors[bone_indices[j,1]-1]) / 2

r"""Plots skeleton pose on a matplotlib axis.

        Args:
            ax (Axis): plt axis to plot
            pose_2d (FloatTensor): tensor of keypoints, of shape K x 2
            bones (list): list of tuples, each tuple defining the keypoint indices to be connected by a bone 
        Returns:
            Module: self
"""
def plot_skeleton(ax, pose_2d, bones=val_data.skeleton, linewidth=2, linestyle='-', label=None, alpha=1):
    cmap = plt.get_cmap('prism')
    for i, bone in enumerate(bones):
        a = 1
        if pose_2d.shape[1] == 3:
            a = pose_2d[bone[0]-1][2].item() * pose_2d[bone[1]-1][2].item() > 0
        elif pose_2d.shape[1] == 6:
            a = min(pose_2d[bone[0]-1][5].item(), pose_2d[bone[1]-1][5].item())
        color = bone_colors[i].tolist()
        if i!=0:
            label=None
        ax.plot(
            (pose_2d[bone[0]-1][0], pose_2d[bone[1]-1][0]),
            (pose_2d[bone[0]-1][1], pose_2d[bone[1]-1][1]),
            linestyle, color=color, linewidth=linewidth, label=label, alpha=alpha * a
        )

r"""Plots list of skeleton poses and image.

        Args:
            poses (list): list of pose tensors to be plotted
            ax (Axis): plt axis to plot
            bones (list): list of tuples, each tuple defining the keypoint indices to be connected by a bone 
        Returns:
            Module: self
"""
def plotPosesOnImage(poses, img, ax=plt, labels=None):
    img_pil = torchvision.transforms.ToPILImage()(img)
    img_size = torch.FloatTensor(img_pil.size)
    linestyles = ['-', ':', '--', '-.']
    for i, p in enumerate(poses):
        pose_px = p
        plot_skeleton(ax, pose_px, linestyle=linestyles[i%len(linestyles)], label=labels[i])
    ax.imshow(img_pil)

def plotMultiPosesOnImage(poses, img, ax=plt, label=None):
    img_pil = torchvision.transforms.ToPILImage()(img)
    img_size = torch.FloatTensor(img_pil.size)
    for i, p in enumerate(poses):
        pose_px = p
        plot_skeleton(ax, pose_px, linestyle='-', label=(label if i==0 else None), alpha=10/len(poses))
    ax.imshow(img_pil)

r"""Converts a multi channel heatmap to an RGB color representation for display.

        Args:
            heatmap (tensor): of size C X H x W
        Returns:
            image (tensor): of size 3 X H x W
"""
def heatmap2image(heatmap):
    C,H,W = heatmap.shape
    img = torch.FloatTensor(3,H,W).fill_(0)
    
    max_ = torch.max(torch.max(heatmap, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
    z = torch.sum(torch.exp(heatmap - max_), (1, 2)).view(C, 1, 1)
    h_norm = torch.exp(heatmap - max_) / z
    presence_prob = 1 - 1 / (z/(H*W)*max_ + 1).view(C, 1, 1)
    h_norm *= presence_prob / torch.max(presence_prob)

    for i in range(C):
        color = joint_colors[i].reshape([-1,1,1])
        img = torch.max(img, color * h_norm[i]) # max in case of overlapping position of joints
    # heatmap and probability maps might have small maximum value. Normalize per channel to make each of them visible
    #img_max, indices = torch.max(img,dim=-1,keepdim=True)
    #img_max, indices = torch.max(img_max,dim=-2,keepdim=True)
    img_max = torch.max(img)
    return img/img_max

def plot_pose_confidence(full_pose, ax=plt):
    i=0
    for joint in full_pose:
        cov_idx = torch.tensor([[2,4],[4,3]])
        cov_mat = joint[cov_idx]
        lambda_, v = np.linalg.eig(cov_mat)
        lambda_ = np.sqrt(lambda_)
        ellipse = Ellipse(xy=(joint[0], joint[1]),
                  width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=np.rad2deg(np.arctan2(*v[:,0][::-1])),
                  color=joint_colors[i].tolist(),
                  alpha=joint[5].item(),
                  lw=1)
        ellipse.set_facecolor('none')
        ax.add_artist(ellipse)
        i+=1
    return None

preds = predictions['pose']
heatmaps = predictions['heatmap']

fig=plt.figure(figsize=(16, 9), dpi= 80, facecolor='w', edgecolor='k')
axes=fig.subplots(1,2)

image_cpu = batch['image'].cpu().detach()
pose_cpu = batch['pose'].cpu().detach()

pred_cpu = preds.cpu().detach()
heatmap_cpu = heatmaps.cpu().detach()

maxmode = torch.min(lf.gaussian_nll(predictions, predictions))
probs = torch.zeros(samples, device="cuda:0")
for i in range(0, samples):
    pred_repeat = {
        'pose': predictions['pose'][i].expand(samples, -1, -1)}
    probs[i] = torch.sum(torch.exp(maxmode-lf.gaussian_nll(predictions, pred_repeat)))
bestpose = torch.argmax(probs)
#bestpose = torch.argmin(lf.gaussian_entropy(predictions))
#bestpose = 0

# plot the ground truth and the predicted poses on top of the image
plotPosesOnImage([pred_cpu[bestpose].detach(), pose_cpu[0]], val_data.denormalize(image_cpu[0])[[2,1,0],:,:], ax=axes[0], labels=['prediction', 'ground truth label'])
axes[0].set_title('Input image with predicted pose (solid) and GT pose (dashed)')
axes[0].legend()

# plot the predicted probability map and the predicted pose on top
plotMultiPosesOnImage(pred_cpu.detach(), heatmap2image(heatmap_cpu[bestpose]).detach(), ax=axes[1], label='predictions')
axes[1].set_title('Predicted probability map with predicted pose overlayed')
axes[1].legend()

plot_pose_confidence(pred_cpu[bestpose].detach(), axes[0])
#plot_pose_confidence(pred_cpu[0].detach(), axes[1])

plt.show()
image_modified = val_data.denormalize(image_cpu[0])
print('done')