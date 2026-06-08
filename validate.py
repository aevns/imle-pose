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
    "mse_1", "mse_m4", "mse_m8", "mse_m16", "mse_s4", "mse_s8", "mse_s16",
    "gauss_1", "gauss_m4", "gauss_m8", "gauss_m16", "gauss_s4", "gauss_s8", "gauss_s16",
    "dkl_1", "dkl_m4", "dkl_m8", "dkl_m16", "dkl_s4", "dkl_s8", "dkl_s16"
]
checkpoint = "network_899.pth"
samples = [
    1, 4, 8, 16, 4, 8, 16,
    1, 4, 8, 16, 4, 8, 16,
    1, 4, 8, 16, 4, 8, 16
]
samples = [
    1, 100, 100, 100, 100, 100, 100,
    1, 100, 100, 100, 100, 100, 100,
    1, 100, 100, 100, 100, 100, 100
]
leg_swaps = 0.5
arm_swaps = 0.1
#loss_functions = [
#    lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse, lf.heatmap_target_mse,
#    lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll, lf.gaussian_nll,
#    lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl
#]
loss_functions = [
    lf.heatmap_mpjpe, lf.heatmap_mpjpe, lf.heatmap_mpjpe, lf.heatmap_mpjpe, lf.heatmap_mpjpe, lf.heatmap_mpjpe, lf.heatmap_mpjpe,
    lf.gaussian_dkl, lf.gaussian_dkl, lf.gaussian_dkl, lf.gaussian_dkl, lf.gaussian_dkl, lf.gaussian_dkl, lf.gaussian_dkl,
    lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl, lf.heatmap_target_dkl
]
sample_methods = [
    "constant", "select", "select", "select", "select", "select", "select",
    "constant", "select", "select", "select", "select", "select", "select",
    "constant", "select", "select", "select", "select", "select", "select"
]
generate_heatmaps = [
    False, False, False, False, False, False, False,
    True, True, True, True, True, True, True,
    True, True, True, True, True, True, True
]

#########################################################################

all_losses = []
for m in range(7, len(folder_names)):

    loss_history = []

    dir = "output/{}/state_dict/".format(folder_names[m])
    #files = os.listdir(dir)
    #files.sort(key = lambda x: int(re.search(r'\d+', x).group()))
    files = [checkpoint]
    e = 0
    for filename in files:
        
        val_data = HDF5Dataset(
            data_file,
            leg_swaps=leg_swaps,
            arm_swaps=arm_swaps,
            generate_heatmaps=True,
            device="cuda:0")
        
        val_sampler = HDF5Sampler(
            data_source=val_data,
            seed=145)
        
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=0,
            sampler=val_sampler
        )
        
        model = network(loss_functions[m], val_data.image_size, noise_length=8).cuda()
        model.train(False)

        f = os.path.join(dir, filename)
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.train(False)
        del state_dict
        epoch_loss_history = []

        val_iter = iter(val_loader)
        val_loss = 0
        for i in range(len(val_loader)):
            batch = {k:v.cuda(0, non_blocking = True) for k, v in next(val_iter).items()}
            
            if sample_methods[m] == "mixed":
                loss = model.mixed_sample_loss(batch, samples[m])
                loss = torch.sum(losses)
            elif sample_methods[m] == "select":
                losses = model.min_sample_loss(batch, samples[m])
                loss = torch.sum(losses)
            elif sample_methods[m] == "constant":
                losses = model.unconditioned_loss(batch)
                loss = torch.sum(losses)
            val_loss += loss.item() / len(val_loader.dataset)
        print("{}, {}, {}".format(folder_names[m], filename, val_loss))

        epoch_loss_history.append(val_loss)
        loss_history.append(epoch_loss_history)
        del val_iter
        del val_loader
        del epoch_loss_history
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