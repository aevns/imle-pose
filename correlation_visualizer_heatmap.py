import argparse
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
from models.unet_pretrained import UNetPretrained

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', '-d', nargs='?', default="data/stick/val.hdf5")
parser.add_argument('--model', '-m', nargs='?', default='UNet', choices=['UNet', 'UNetPretrained'])
parser.add_argument('--folder', '-f', nargs='?', default="dkl_s16")
parser.add_argument('--checkpoint', '-cp', nargs='?', default="network_899.pth")
parser.add_argument('--loss', '-l', nargs='?', default='dkl', choices=['mse', 'gaussian', 'dkl'])
parser.add_argument('--samples', '-sm', nargs='?', default=100, type=int)
parser.add_argument('--legswaps', '-ls', nargs='?', default=0.0, type=float)
parser.add_argument('--armswaps', '-as', nargs='?', default=0.0, type=float)
parser.add_argument('--seed', nargs='?', default=26, type=int)
parser.add_argument('--joints', '-j', nargs=2, default=[11, 14], type=int)
parser.add_argument('--coords', '-c', nargs=2, default=['x', 'x'], choices=['x', 'y'])
parser.add_argument('--save', nargs='?', default=None)
args = parser.parse_args()

data_file = args.datafile
network = UNet if args.model == 'UNet' else UNetPretrained
folder_name = args.folder
checkpoint = args.checkpoint
samples = args.samples
leg_swaps = args.legswaps
arm_swaps = args.armswaps
loss_function = {'mse': lf.heatmap_target_mse, 'gaussian': lf.gaussian_nll, 'dkl': lf.heatmap_target_dkl}[args.loss]
flipped_joints = [0, 1, 2, 3, 4, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
plot_variables = torch.tensor([[j, 'xy'.index(c)] for j, c in zip(args.joints, args.coords)])
opposite_variables = torch.tensor([[flipped_joints[j], c] for j, c in plot_variables.tolist()])

val_data = HDF5Dataset(
    data_file,
    leg_swaps = leg_swaps,
    arm_swaps = arm_swaps,
    generate_heatmaps=True,
    device="cuda:0")

val_sampler = HDF5Sampler(
    data_source=val_data,
    seed=args.seed)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = 1,
    num_workers = 0,
    sampler=val_sampler)

model = network(loss_function, val_data.image_size, 8).cuda()
model.train(False)

state_dict = torch.load("output/{}/state_dict/{}".format(folder_name, checkpoint))
model.load_state_dict(state_dict)

with torch.no_grad():
    val_iter = iter(val_loader)
    batch = next(val_iter)
    batch_repeat = {'image': batch['image'].expand(samples, -1, -1, -1)}
    predictions = model.get_sample(batch_repeat)

pose_cpu = batch['pose'][0].cpu().detach()

size_x, size_y = [val_data.image_size[1 - c].item() for c in plot_variables[:,1]]
img = torch.zeros((3,size_y,size_x))

(j1, c1), (j2, c2) = plot_variables.tolist()
for i in range(samples):
    heatmap_cpu = predictions['heatmap'][i].cpu().detach()

    if args.loss == 'mse':
        # MSE heatmaps are unnormalized activations
        heatmap_cpu = torch.clamp(heatmap_cpu, min=0)
        heatmap_cpu = heatmap_cpu / torch.sum(heatmap_cpu, (1, 2), keepdim=True)
    else:
        C,H,W = heatmap_cpu.shape
        max_ = torch.max(torch.max(heatmap_cpu, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
        z = torch.sum(torch.exp(heatmap_cpu - max_), (1, 2)).view(C, 1, 1)
        heatmap_cpu = torch.exp(heatmap_cpu - max_) / z

    # marginal distribution of each plotted coordinate
    x_vals = torch.sum(heatmap_cpu[j1],dim=c1)
    y_vals = torch.sum(heatmap_cpu[j2],dim=c2)

    img[:] += torch.outer(y_vals, x_vals)

img = img - torch.min(img)
img = img / torch.max(img)
img_pil = torchvision.transforms.ToPILImage()(img)

def axis_label(joint, coord):
    name = val_data.keypoints[joint].decode().replace('_', ' ').title()
    return "{} Position of {} (px)".format(['Horizontal', 'Vertical'][coord], name)

fig=plt.figure(constrained_layout=True, figsize=(3, 3), dpi= 300, facecolor='w', edgecolor='k')
axes=fig.subplots(1,1)
axes.imshow(img_pil, extent=(0, size_x, size_y, 0))

axes.set_xlabel(axis_label(j1, c1))
axes.set_ylabel(axis_label(j2, c2))
axes.plot(pose_cpu[plot_variables[0,0],plot_variables[0,1]]+0.5, pose_cpu[plot_variables[1,0],plot_variables[1,1]]+0.5, color="red", marker="x", markersize=4)
axes.plot(pose_cpu[opposite_variables[0,0],opposite_variables[0,1]]+0.5, pose_cpu[opposite_variables[1,0],opposite_variables[1,1]]+0.5, color="cyan", marker="x", markersize=4)

if args.save:
    plt.savefig(args.save)
else:
    plt.show()
print('done')
