import torch
import torch.nn as nn
import os

#from tqdm import tqdm

from dataset import HDF5Dataset
from models.basic import Basic
from models.basic_vector import BasicVector
from models.basic_progressive import BasicProgressive
from models.basic_spatial import BasicSpatial

#########################################################################

start_epoch = 0
num_epochs = 80

data_file = "./data/stick/train.hdf5"
swap_rate = 0.5

model_name = "spatial_swaps"
model = BasicSpatial
implicit = True
samples = 16

#########################################################################

os.makedirs(os.path.dirname("output/{}/state_dict/".format(model_name)), exist_ok=True)

train_data = HDF5Dataset((64,48), swap_rate, data_file)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

network = model().cuda()
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

        loss = 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, batch['target']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i%(len(train_loader)//10)==0:
            print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), losses[-1]))
    torch.save(network.state_dict(), "output/{}/state_dict/network_{}.pth".format(model_name, e))