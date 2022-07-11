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

import models.utils.loss_functions as lf

from dataset import HDF5Dataset
from models.simple18 import SimplePose
from models.basic import Basic
from models.basic_vector import BasicVector
from models.basic_progressive import BasicProgressive
from models.basic_spatial import BasicSpatial
from models.unet import UNet
from models.unet_vector import UNetVector
from models.dcignet import DCIGNet

val_data = HDF5Dataset("./data/stick/val.hdf5", 0.5)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=False,
    drop_last=True
)

network = UNetVector(loss_function = lf.heatmap_gaussian_fit_entropy).cuda()

state_dict = torch.load("./output/unet_vector_gaussian_swaps/state_dict/network_79.pth")
network.load_state_dict(state_dict)
network.training = False

val_iter = iter(val_loader)
for i in range(1):
    batch = next(val_iter)
pred, z = network.sample(batch)

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
        color = cmap((bone[1]-1) * cmap.N // len(val_data.keypoints)) # color according to second joint index
        if i!=0:
            label=None
        ax.plot(
            (pose_2d[bone[0]-1][0], pose_2d[bone[1]-1][0]),
            (pose_2d[bone[0]-1][1], pose_2d[bone[1]-1][1]),
            linestyle, color=color, linewidth=linewidth, label=label
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
    linestyles = ['-', '--', '-.', ':']
    for i, p in enumerate(poses):
        pose_px = p
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

#pred_pose = torch.flatten(pred_cpu, start_dim=2)
#pred_pose = torch.argmax(pred_pose, dim=2)
#pred_pose = torch.stack([pred_pose % pred_cpu.shape[3], pred_pose // pred_cpu.shape[3]], -1)
#pred_heatmaps = pred_cpu

pred_pose = lf.heatmap_gaussian_fit(pred_cpu)[:,:,0:2]
pred_heatmaps = lf.heatmaps_normalized(pred_cpu)

# plot the ground truth and the predicted pose on top of the image
plotPosesOnImage([pred_pose[0].detach(), pose_cpu[0]], val_data.denormalize(image_cpu[0]), ax=axes[0], labels=['prediction', 'ground truth label'])
axes[0].set_title('Input image with predicted pose (solid) and GT pose (dashed)')
axes[0].legend()

# plot the predicted probability map and the predicted pose on top
plotPosesOnImage([pred_pose[0].detach()], heatmap2image(pred_heatmaps[0]).detach(), ax=axes[1], labels=['prediction'])
axes[1].set_title('Predicted probability map with predicted pose overlayed')
axes[1].legend()

plt.show()
image_modified = val_data.denormalize(image_cpu[0])
print('done')