from cProfile import label
import json
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import h5py
import os
import re

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import cv2

from torch import dtype, uint8
from tqdm import tqdm
from collections import defaultdict

from dataset import HDF5Dataset
from models.basic import Basic
from models.basic_vector import BasicVector
from models.basic_progressive import BasicProgressive
from models.basic_spatial import BasicSpatial

#########################################################################

data_file = "./data/stick/val.hdf5"
swap_rate = 0.5

model_names = ["basic_swaps", "vector_swaps", "progressive_swaps", "spatial_swaps"]
models = [Basic, BasicVector, BasicProgressive, BasicSpatial]
implicit = [False, True, True, True]
samples = [0, 5, 5, 5]

#########################################################################

val_data = HDF5Dataset((64,48), swap_rate, data_file)

all_losses = []
for m in range(len(model_names)):
    network = models[m]().cuda()
    network.training = False
    losses = []

    dir = "output/{}/state_dict/".format(model_names[m])
    files = os.listdir(dir)
    files.sort(key = lambda x: int(re.search(r'\d+', x).group()))

    e = 0
    for filename in files:
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=128,
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
                pred, _ = network.train_sample(batch, samples[m])
            else:
                pred = network(batch)

            loss = 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, batch['target']))
            epoch_losses.append(loss.item())

        losses.append(np.average(epoch_losses))
        print("losses for model {}, epoch {}: {}".format(m, e, losses[e]))
        e += 1
    all_losses.append(losses)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_yscale('log')
for i in range(len(all_losses)):
    ax.plot(all_losses[i], label=model_names[i])
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()
plt.show()