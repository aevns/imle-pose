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
model_name = "unet_heatmap_swaps"
epoch = 36
samples=1
plot_variables = torch.tensor([[11,0],[14,0]])

val_data = HDF5Dataset("./data/stick/val.hdf5", 0, (64,48))

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=False,
    drop_last=True
)

network = model(loss_function = lf.heatmap_target_mse).cuda()

state_dict = torch.load("./output/{}/state_dict/network_{}.pth".format(model_name, epoch))
network.load_state_dict(state_dict)
network.training = False

val_iter = iter(val_loader)
for i in range(1): # 87, 101
    batch = next(val_iter)

#image_cpu = batch['image'][0].cpu().detach()
heatmap_cpu = batch['target'][0].cpu().detach()
pose_cpu = batch['pose'][0].cpu().detach()

img = torch.zeros((3,64,64))

for i in range(samples):
    predictions = network.sample(batch)
    heatmap = predictions['heatmap'][0]
    heatmap_cpu = heatmap.cpu().detach()
    
    x_vals = torch.sum(heatmap_cpu[plot_variables[0,0]],dim=plot_variables[0,1].item())
    y_vals = torch.sum(heatmap_cpu[plot_variables[1,0]],dim=plot_variables[1,1].item())

    img[:,:len(y_vals),:len(x_vals)] += torch.outer(y_vals, x_vals)

img = img - torch.min(img)
img = img / torch.max(img)
img_pil = torchvision.transforms.ToPILImage()(img)

fig=plt.figure(figsize=(16, 9), dpi= 80, facecolor='w', edgecolor='k')
axes=fig.subplots(1,1)
axes.imshow(img_pil)
axes.set_xlabel("Position of joint {}".format(plot_variables[0]))
axes.set_ylabel("Position of joint {}".format(plot_variables[1]))

axes.plot([0,64], [0,64])
axes.plot(pose_cpu[plot_variables[0,0],plot_variables[0,1]], pose_cpu[plot_variables[1,0],plot_variables[1,1]], marker="o")
axes.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, pos: x*8))
axes.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, pos: y*8))
plt.show()
print('done')