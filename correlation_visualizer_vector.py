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

import loss_functions as lf

from dataset import HDF5Dataset
from models.unet import UNet
from models.unet_vector import UNetVector

model = UNet
model_name = "unet_gaussian_swaps"
epoch = 41
samples=100
image_scale = 8
plot_variables = torch.tensor([[11,0],[14,0]])

val_data = HDF5Dataset("./data/stick/val.hdf5", 0)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=False,
    drop_last=True
)

network = model(loss_function = lf.gaussian_nll).cuda()

state_dict = torch.load("./output/{}/state_dict/network_{}.pth".format(model_name, epoch))
network.load_state_dict(state_dict)
network.training = False

val_iter = iter(val_loader)
for i in range(1): # 87, 101
    batch = next(val_iter)

#image_cpu = batch['image'][0].cpu().detach()
pose_cpu = batch['pose'][0].cpu().detach()

img = torch.zeros((3,64*image_scale,64*image_scale))

for i in range(samples):
    predictions = network.sample(batch)
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
    
    x_vals = torch.linspace(0, 64-1/image_scale, 64*image_scale, device=pred_cpu.device).unsqueeze(0).expand(64*image_scale, 64*image_scale)
    y_vals = torch.linspace(0, 64-1/image_scale, 64*image_scale, device=pred_cpu.device).unsqueeze(1).expand(64*image_scale, 64*image_scale)

    points = torch.stack((x_vals, y_vals), dim=-1)

    dif = (points - mu).unsqueeze(-1)
    gaussian = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(sigma), dif)).squeeze()
    gaussian = torch.det(2*pi*sigma)**(-1/2) * torch.exp(-gaussian/2)
    img[:] += gaussian

img = img / torch.max(img)
img_pil = torchvision.transforms.ToPILImage()(img)

fig=plt.figure(figsize=(16, 9), dpi= 80, facecolor='w', edgecolor='k')
axes=fig.subplots(1,1)
axes.imshow(img_pil)
axes.set_xlabel("Position of joint {}".format(plot_variables[0]))
axes.set_ylabel("Position of joint {}".format(plot_variables[1]))

axes.plot([0,image_scale * 64], [0,image_scale * 64])
axes.plot(pose_cpu[plot_variables[0,0],plot_variables[0,1]]*image_scale, pose_cpu[plot_variables[1,0],plot_variables[1,1]]*image_scale, marker="o")

plt.show()
print('done')