import json
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import h5py

import torchvision.transforms as transforms
import cv2

from torch import dtype, uint8
from tqdm import tqdm
from collections import defaultdict

from dataset import StickDataset
from models.basic import Basic
from models.simple18 import SimplePose

#########################################################################

num_epochs = 40
train_data = StickDataset((29,21),data_file="./data/stick/train.hdf5")
val_data = StickDataset((29,21),data_file="./data/stick/val.hdf5")

#########################################################################

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

network = Basic().cuda()

losses = []
optimizer = torch.optim.Adam(network.parameters(), lr = 0.0001)
for e in range(num_epochs):
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        batch = next(train_iter)
        pred = network(batch)
        loss = 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, batch['target']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i%(len(train_loader)//20)==0:
            print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))
    torch.save(network.state_dict(), "output/basic/state_dict/network_{}.pth".format(e))