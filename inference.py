import json
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import cv2

from torch import dtype, uint8
from tqdm import tqdm
from collections import defaultdict

from dataset import StickDataset
from models.simple18 import SimplePose

val_data = StickDataset((16,16),data_file="./data/stick/val.hdf5")

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

network = SimplePose().cuda()

state_dict = torch.load("./output/simple18/state_dict/network_39.pth")
network.load_state_dict(state_dict)

val_iter = iter(val_loader)
batch = next(val_iter)
pred = network(batch)

# plotting utility functions

r"""Plots skeleton pose on a matplotlib axis.

        Args:
            ax (Axis): plt axis to plot
            pose_2d (FloatTensor): tensor of keypoints, of shape K x 2
            bones (list): list of tuples, each tuple defining the keypoint indices to be connected by a bone 
        Returns:
            Module: self
"""
def plot_skeleton(ax, pose_2d, bones=val_data.skeleton, linewidth=2, linestyle='-', label=None):
    cmap = plt.get_cmap('hsv')
    for i, bone in enumerate(bones):
        color = cmap(bone[1] * cmap.N // len(val_data.keypoints)) # color according to second joint index
        if i!=0:
            label=None
        ax.plot(pose_2d[bone[0]-1], pose_2d[bone[1]-1], linestyle, color=color, linewidth=linewidth, label=label)

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
    linestyles = ['-', '--', '-.', ':']
    for i, p in enumerate(poses):
        pose_px = p*img_size
        plot_skeleton(ax, pose_px, linestyle=linestyles[i%len(linestyles)], label=labels[i])
    ax.imshow(img_pil)

r"""Converts a multi channel heatmap to an RGB color representation for display.

        Args:
            heatmap (tensor): of size C X H x W
        Returns:
            image (tensor): of size 3 X H x W
"""
def heatmap2image(heatmap):
    C,H,W = heatmap.shape
    cmap = plt.get_cmap('hsv')
    img = torch.FloatTensor(3,H,W).fill_(0)
    for i in range(C):
        color = torch.FloatTensor(cmap(i * cmap.N // C)[:3]).reshape([-1,1,1])
        img = torch.max(img, color * heatmap[i]) # max in case of overlapping position of joints
    # heatmap and probability maps might have small maximum value. Normalize per channel to make each of them visible
    img_max, indices = torch.max(img,dim=-1,keepdim=True)
    img_max, indices = torch.max(img_max,dim=-2,keepdim=True)
    return img/img_max

fig=plt.figure(figsize=(20, 5), dpi= 80, facecolor='w', edgecolor='k')
axes=fig.subplots(1,2)

image_cpu = batch['image'].cpu().detach()
pose_cpu = batch['pose'].cpu().detach()
pred_cpu = pred.cpu().detach()

pred_pose = torch.flatten(pred_cpu, start_dim=2)
pred_pose = torch.argmax(pred_pose, dim=2)
pred_pose = torch.stack([pred_pose // pred_cpu.shape[2], pred_pose % pred_cpu.shape[2]], -1)

# plot the ground truth and the predicted pose on top of the image
plotPosesOnImage([pred_pose[0].detach(), pose_cpu[0]], val_data.denormalize(image_cpu[0]), ax=axes[0], labels=['prediction', 'ground truth label'])
axes[0].set_title('Input image with predicted pose (solid) and GT pose (dashed)')
axes[0].legend()

# plot the predicted probability map and the predicted pose on top
plotPosesOnImage([pred_pose[0].detach()], heatmap2image(pred_cpu[0]).detach(), ax=axes[1], labels=['prediction'])
axes[1].set_title('Predicted probability map with predicted pose overlayed')
axes[1].legend()

plt.show()

print('done')