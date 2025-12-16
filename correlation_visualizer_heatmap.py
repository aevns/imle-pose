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
#from models.unet_vector import UNetVector

model = UNet
#model_name = "unet_heatmap_swaps"
model_name = "stick_7"
epoch = 399
samples=1
image_scale = 16
plot_variables = torch.tensor([[8,0],[10,0]])

val_data = HDF5Dataset("data/stick/val.hdf5", generate_heatmaps=True, device="cuda:0")

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

network = model(lf.heatmap_target_mse, val_data.image_size).cuda()
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
    'target': val_data._target_generator(torch.tensor(val_data.hdf5['poses'][n], dtype=torch.float, device=val_data.device)),
    }

#image_cpu = batch['image'][0].cpu().detach()
heatmap_cpu = batch['target'][0].cpu().detach()
pose_cpu = batch['pose'][0].cpu().detach()

img = torch.zeros((3,64,64))

for i in range(samples):
    predictions = network.get_sample(batch)
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
axes.imshow(img_pil, extent=(0, input_size, input_size, 0))
#axes.set_xlabel("Position of joint {}".format(plot_variables[0]))
#axes.set_ylabel("Position of joint {}".format(plot_variables[1]))
axes.set_xlabel("Horizontal Position of Left Elbow (px)")
axes.set_ylabel("Horizontal Position of Left Wrist (px)")
axes.plot(pose_cpu[plot_variables[0,0],plot_variables[0,1]], pose_cpu[plot_variables[1,0],plot_variables[1,1]], marker="o")

plt.show()
print('done')