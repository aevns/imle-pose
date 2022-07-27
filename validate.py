import numpy as np
import torch
import os
import re

import matplotlib.pyplot as plt

import loss_functions as lf

from dataset import HDF5Dataset
from models.unet import UNet
from models.unet_vector import UNetVector

#########################################################################

data_file = "./data/stick/val.hdf5"

models = [UNetVector, UNetVector, UNetVector, UNetVector, UNetVector]
model_names = ["unet_vector_gaussian_swaps", "unet_vector_gaussian_swaps_soft"]
implicit = [True, True]
samples = [20, 20]
swap_rates = [0.5, 0.5]
loss_function = lf.gaussian_nll

#########################################################################

all_losses = []
for m in range(len(model_names)):
    val_data = HDF5Dataset(data_file, swap_rates[m])

    network = models[m](loss_function = loss_function).cuda()
    network.training = False
    losses = []

    dir = "output/{}/state_dict/".format(model_names[m])
    files = os.listdir(dir)
    files.sort(key = lambda x: int(re.search(r'\d+', x).group()))

    e = 0
    for filename in files:
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=64,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
            drop_last=True
        )
        val_iter = iter(val_loader)

        f = os.path.join(dir, filename)
        state_dict = torch.load(f)
        network.load_state_dict(state_dict)
        network.training = False

        epoch_losses = []

        for i in range(len(val_loader)):
            batch = next(val_iter)
            
            if implicit[m]:
                pred = network.sample(batch, samples[m])
            else:
                pred = network(batch)

            loss = torch.mean(network.loss(pred, batch))
            epoch_losses.append(loss.item())

        losses.append(np.average(epoch_losses))
        print("losses for model {}, epoch {}: {}".format(m, e, losses[e]))
        e += 1
    all_losses.append(losses)

fig = plt.figure()
ax = fig.add_subplot()
#ax.set_yscale('log')
for i in range(len(all_losses)):
    if implicit[i]:
        ax.plot(all_losses[i], label="{} ({} samples)".format(model_names[i], samples[i]))
    else:
        ax.plot(all_losses[i], label="{}".format(model_names[i]))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (Entropy)')
ax.legend()
plt.show()