import torch
import torch.nn as nn
import os

#from tqdm import tqdm

import models.utils.loss_functions as lf

from dataset import HDF5Dataset
from models.unet import UNet
from models.unet_vector import UNetVector

#########################################################################

start_epoch = 40
num_epochs = 80

data_file = "./data/stick/train.hdf5"
swap_rate = 0.5

model_name = "unet_gaussian_swaps"
model = UNet
implicit = False
samples = 6
loss_function = lf.heatmap_gaussian_fit_entropy

#########################################################################

os.makedirs(os.path.dirname("output/{}/state_dict/".format(model_name)), exist_ok=True)

train_data = HDF5Dataset(data_file, swap_rate, (64, 48))

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

network = model(loss_function=loss_function).cuda()
if start_epoch > 0:
    state_dict = torch.load("./output/{}/state_dict/network_{}.pth".format(model_name, start_epoch - 1))
    network.load_state_dict(state_dict)
network.training = True

losses = []
optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
for e in range(start_epoch, num_epochs):
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        batch = next(train_iter)

        if implicit:
            pred, _ = network.train_sample(batch, samples)
        else:
            pred = network(batch)

        loss = torch.mean(network.loss(pred, batch))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i%(len(train_loader)//10)==0:
            print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))
    torch.save(network.state_dict(), "output/{}/state_dict/network_{}.pth".format(model_name, e))