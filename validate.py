import numpy as np
import torch
import os
import re

import matplotlib.pyplot as plt

import models.utils.loss_functions as lf

from dataset import HDF5Dataset
from models.unet import UNet

#########################################################################

data_file = "data/stick/val.hdf5"
batch_size = 32

models = [UNet]
folder_names = ["model_2"]
noise_lengths = [8]
samples = [20]
swap_rates = [0.5]
loss_function = lf.gaussian_nll
target_heatmaps = [False]
sample_methods = ["mixed"]

#########################################################################

all_losses = []
for m in range(len(folder_names)):
    val_data = HDF5Dataset(data_file, swap_rates[m], target_heatmaps[m])

    network = models[m](loss_function, val_data.image_size, noise_lengths[m]).cuda()
    network.training = False
    loss_history = []

    dir = "output/{}/state_dict/".format(folder_names[m])
    files = os.listdir(dir)
    files.sort(key = lambda x: int(re.search(r'\d+', x).group()))

    e = 0
    for filename in files:
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
            drop_last=True
        )
        val_iter = iter(val_loader)

        f = os.path.join(dir, filename)
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        network.load_state_dict(state_dict)
        network.training = False
        del state_dict
        epoch_loss_history = []

        for i in range(len(val_loader)):
            batch = next(val_iter)
            
            if sample_methods[m] == "mixed":
                losses = network.mixed_sample_loss(batch, samples[m])
            elif sample_methods[m] == "min":
                losses = network.min_sample_loss(batch, samples[m])
            else:
                losses = network.loss(network(batch), batch)

            loss = torch.mean(losses)
            del losses
            epoch_loss_history.append(loss.cpu().detach().item())
            del loss
        del val_iter
        del val_loader
        loss_history.append(np.average(epoch_loss_history))
        del epoch_loss_history
        print("losses for model {}, epoch {}: {}".format(m, e, loss_history[e]))
        e += 1
    all_losses.append(loss_history)
    del loss_history

fig = plt.figure()
ax = fig.add_subplot()
#ax.set_yscale('log')
for i in range(len(all_losses)):
    ax.plot(all_losses[i], label="{} ({} samples)".format(folder_names[i], samples[i]))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (Entropy)')
ax.legend()
plt.show()