import numpy as np
import torch
import os
import re

import matplotlib.pyplot as plt

import models.utils.loss_functions as lf
from dataset import HDF5Sampler
from dataset import HDF5Dataset
from models.unet import UNet, UNetLarge

#########################################################################

data_file = "data/stick/val.hdf5"
batch_size = 32

network = UNetLarge
folder_names = [
    "mse_1L", "mse_m4L", "mse_m8L", "mse_s4L", "mse_s8L",
    "gauss_1L", "gauss_m4L", "gauss_m8L", "gauss_s4L", "gauss_s8L",
    "dkl_1L", "dkl_m4L", "dkl_m8L", "dkl_s4L", "dkl_s8L"
]
checkpoint = "network_599.pth"
samples = 40
leg_swaps = 0.5
arm_swaps = 0.1
loss_functions = [
    lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse,
    lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll,
    lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl
]
sample_methods = [
    "constant", "mixed", "mixed", "select", "select",
    "constant", "mixed", "mixed", "select", "select",
    "constant", "mixed", "mixed", "select", "select"
]
generate_heatmaps = [
    False, False, False, False, False,
    True, True, True, True, True,
    True, True, True, True, True
]

#########################################################################

all_losses = []
for m in range(len(folder_names)):

    val_data = HDF5Dataset(data_file, leg_swaps, arm_swaps, generate_heatmaps=True, device="cuda:0")
    val_sampler = HDF5Sampler(data_source=val_data)
    model = network(loss_functions[m], val_data.image_size, noise_length=8).cuda()
    model.training = False
    loss_history = []

    dir = "output/{}/".format(folder_names[m])
    files = os.listdir(dir)
    files.sort(key = lambda x: int(re.search(r'\d+', x).group()))

    e = 0
    for filename in files:

        f = os.path.join(dir, filename)
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.training = False
        del state_dict
        epoch_loss_history = []

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=0,
            #pin_memory=False,
            #shuffle=True,
            #drop_last=True,
            sampler=val_sampler
        )
        val_iter = iter(val_loader)
        val_loss = 0
        for i in range(len(val_loader)):
            batch = {k:v.cuda(0, non_blocking = True) for k, v in next(val_iter).items()}
            
            if sample_methods[m] == "mixed":
                loss = model.mixed_sample_loss(batch, samples)
            elif sample_methods[m] == "select":
                losses = model.min_sample_loss(batch, samples)
                loss = torch.mean(losses)
            elif sample_methods[m] == "constant":
                losses = model.unconditioned_loss(batch)
                loss = torch.mean(losses)
            val_loss += loss.item()
        epoch_loss_history.append(val_loss / len(val_loader))
        
        loss_history.append(epoch_loss_history)
        del val_iter
        del val_loader
        del epoch_loss_history
        print("losses for model {}, epoch {}: {}".format(m, e, loss_history[e]))
        e += 1
    all_losses.append(loss_history)
    del loss_history

fig = plt.figure()
ax = fig.add_subplot()
#ax.set_yscale('log')
for i in range(len(all_losses)):
    ax.plot(all_losses[i], label="{}".format(folder_names[i]))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (Entropy)')
ax.legend()
plt.show()