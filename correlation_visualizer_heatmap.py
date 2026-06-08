import json
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import h5py
from math import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torchvision
import torchvision.transforms as transforms
import cv2

from torch import dtype, uint8
from tqdm import tqdm
from collections import defaultdict

import models.utils.loss_functions as lf

from dataset import HDF5Sampler
from dataset import HDF5Dataset
from models.unet import UNet, UNetLarge

data_file = "data/stick/val.hdf5"
network = UNetLarge
folder_name = "dkl_s16"
checkpoint = "network_899.pth"
samples = 100
leg_swaps = 0.0 #0.5
arm_swaps = 0.0 #0.1
loss_function = lf.heatmap_target_dkl
sample_method = "select"
image_scale = 16
plot_variables = torch.tensor([[11,0],[14,0]])
opposite_variables = torch.tensor([[12,0],[13,0]])

val_data = HDF5Dataset(
    data_file,
    leg_swaps = leg_swaps,
    arm_swaps = arm_swaps,
    generate_heatmaps=True,
    device="cuda:0")

val_sampler = HDF5Sampler(
    data_source=val_data,
    seed=26)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = 1,
    num_workers = 0,
    sampler=val_sampler)

model = network(loss_function, val_data.image_size, 8).cuda()
model.train(False)
input_size = model.input_size[1]

state_dict = torch.load("output/{}/state_dict/{}".format(folder_name, checkpoint))
model.load_state_dict(state_dict)
model.training = False

with torch.no_grad():
    val_iter = iter(val_loader)
    batch = next(val_iter)
    batch_repeat = {'image': batch['image'].expand(samples, -1, -1, -1)}
    predictions = model.get_sample(batch_repeat)

pose_cpu = batch['pose'][0].cpu().detach()

img = torch.zeros((3,48,48))

for i in range(samples):
    predictions = model.get_sample(batch)
    heatmap = predictions['heatmap'][0]
    heatmap_cpu = heatmap.cpu().detach()
    
    C,H,W = heatmap_cpu.shape
    max_ = torch.max(torch.max(heatmap_cpu, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
    z = torch.sum(torch.exp(heatmap_cpu - max_), (1, 2)).view(C, 1, 1)
    heatmap_cpu = torch.exp(heatmap_cpu - max_) / z

    x_vals = torch.sum(heatmap_cpu[plot_variables[0,0]],dim=plot_variables[0,1].item())
    y_vals = torch.sum(heatmap_cpu[plot_variables[1,0]],dim=plot_variables[1,1].item())

    img[:,:len(y_vals),:len(x_vals)] += torch.outer(y_vals, x_vals)

img = img - torch.min(img)
img = img / torch.max(img)
img_pil = torchvision.transforms.ToPILImage()(img)

fig=plt.figure(constrained_layout=True, figsize=(3, 3), dpi= 300, facecolor='w', edgecolor='k')
axes=fig.subplots(1,1)
axes.imshow(img_pil, extent=(0, input_size, input_size, 0))

axes.set_xlabel("Horizontal Position of Left Hip (px)")
axes.set_ylabel("Horizontal Position of Right Knee (px)")
axes.plot(pose_cpu[plot_variables[0,0],plot_variables[0,1]]+0.5, pose_cpu[plot_variables[1,0],plot_variables[1,1]]+0.5, color="red", marker="x", markersize=4)
axes.plot(pose_cpu[opposite_variables[0,0],opposite_variables[0,1]]+0.5, pose_cpu[opposite_variables[1,0],opposite_variables[1,1]]+0.5, color="cyan", marker="x", markersize=4)

plt.show()
print('done')