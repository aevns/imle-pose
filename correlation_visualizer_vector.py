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
from models.unet import UNet

model = UNet
model_name = "old/"
epoch = 12
samples=100
image_scale = 16
plot_variables = torch.tensor([[11,0],[14,0]])

val_data = HDF5Dataset("data/simple/val.hdf5", generate_heatmaps=False, device="cuda:0")

val_sampler = HDF5Sampler(
    data_source=val_data)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = 1,
    num_workers=0,
    sampler=val_sampler,
    pin_memory=True,
    shuffle=False,
    drop_last=True)

network = model(lf.gaussian_nll, val_data.image_size, 8).cuda()
input_size = network.input_size[1]

state_dict = torch.load("./output/{}/state_dict/network_{}.pth".format(model_name, epoch))
network.load_state_dict(state_dict)
network.training = False

#val_iter = iter(val_loader)
#for i in range(40): # 87, 101
#    batch = next(val_iter)
n = 40
batch = {
    'image': val_data.normalize(torch.tensor(val_data.hdf5['images'][n], dtype=torch.float, device=val_data.device))[None, :, :, :],
    'pose': torch.tensor(val_data.hdf5['poses'][n], dtype=torch.float, device=val_data.device)[None, :, :],
    }

pose_cpu = batch['pose'][0].cpu().detach()

img = torch.zeros((3,input_size*image_scale,input_size*image_scale))

for i in range(samples):
    predictions = network.get_sample(batch)
    pred = predictions['pose'][0]
    pred_cpu = pred.cpu().detach()
    heatmap = predictions['heatmap'][0]
    heatmap_cpu = heatmap.cpu().detach()

    j1 = pred_cpu[plot_variables[0,0],plot_variables[0,1]]
    j2 = pred_cpu[plot_variables[1,0],plot_variables[1,1]]
    mu = torch.stack((j1, j2))
    cov_idx = torch.tensor([[2,4],[4,3]])
    if (plot_variables[0,0] == plot_variables[1,0]):
        sigma = pred_cpu[plot_variables[0,0],cov_idx]
    else:
        sigma = torch.tensor([
            [pred_cpu[plot_variables[0,0],plot_variables[0,1]+2], 0],
            [0, pred_cpu[plot_variables[1,0],plot_variables[1,1]+3]]
            ])
    
    x_vals = torch.linspace(0, input_size-1/image_scale, input_size*image_scale, device=pred_cpu.device).unsqueeze(0).expand(input_size*image_scale, input_size*image_scale)
    y_vals = torch.linspace(0, input_size-1/image_scale, input_size*image_scale, device=pred_cpu.device).unsqueeze(1).expand(input_size*image_scale, input_size*image_scale)

    points = torch.stack((x_vals, y_vals), dim=-1)

    dif = (points - mu).unsqueeze(-1)
    gaussian = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(sigma), dif)).squeeze()
    gaussian = torch.det(2*pi*sigma)**(-1/2) * torch.exp(-gaussian/2)
    img[:] += gaussian

img = img / torch.max(img)
img_pil = torchvision.transforms.ToPILImage()(img)

fig=plt.figure(figsize=(16, 9), dpi= 80, facecolor='w', edgecolor='k')
axes=fig.subplots(1,1)
axes.imshow(img_pil, extent=(0, input_size, input_size, 0))
#axes.set_xlabel("Position of joint {}".format(plot_variables[0]))
#axes.set_ylabel("Position of joint {}".format(plot_variables[1]))
axes.set_xlabel("Horizontal Position of Left Elbow (px)")
axes.set_ylabel("Horizontal Position of Left Wrist (px)")
axes.plot(pose_cpu[plot_variables[0,0],plot_variables[0,1]], pose_cpu[plot_variables[1,0],plot_variables[1,1]], marker=".")

plt.show()
print('done')