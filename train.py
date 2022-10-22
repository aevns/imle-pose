import argparse

import os
import random

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import models.utils.loss_functions as lf
from dataset import HDF5Dataset
from models.unet import UNet
from models.unet_pretrained import UNetPretrained

import wandb

#########################################################################
print(torch.cuda.get_arch_list())
print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', '-d')
parser.add_argument('--batchsize', '-b', nargs='?', default=64, type=int)
parser.add_argument('--model', '-m', nargs='?', default='UNet')
parser.add_argument('--start', '-s', nargs='?', default=0, type=int)
parser.add_argument('--checkpoints', '-cp', nargs='?', default=1, type=int)
parser.add_argument('--end', '-e', nargs='?', default=400, type=int)
parser.add_argument('--loss', '-l', nargs='?', default='gaussian')
parser.add_argument('--samples', '-sm', nargs='?', default=20, type=int)
parser.add_argument('--combine', '-c', nargs='?', default='select')
parser.add_argument('--armswaps', '-as', nargs='?', default=0, type=float)
parser.add_argument('--legswaps', '-ls', nargs='?', default=0, type=float)
parser.add_argument('--output', '-o', nargs='?', default='unnamed')
args = parser.parse_args()
torch.autograd.set_detect_anomaly(False)

start_epoch = args.start
end_epoch = args.end
checkpoint_freq = (end_epoch - start_epoch) / args.checkpoints
batch_size = args.batchsize

train_data_filename = args.dataroot + "train.hdf5"
val_data_filename = args.dataroot + "val.hdf5"

leg_swaps = args.legswaps
arm_swaps = args.armswaps

# Run parameters
wandb.init(
    project="imle-pose",
    config = {'args': args})

if args.model == 'UNet':
    network = UNet
elif args.model == 'UNetPretrained':
    network = UNetPretrained
noise_length = 8

samples = args.samples
if args.loss == 'gaussian':
    loss_function = lf.gaussian_nll
    generate_heatmaps = False
elif args.loss == 'mse':
    loss_function = lf.heatmap_target_mse
    generate_heatmaps = True
elif args.loss == 'dkl':
    loss_function = lf.heatmap_target_dkl
    generate_heatmaps = True

sample_method = args.combine

output_folder = args.output
#########################################################################

os.makedirs(os.path.dirname("output/{}/state_dict/".format(output_folder)), exist_ok=True)
os.makedirs(os.path.dirname("output/{}/training_log/".format(output_folder)), exist_ok=True)

train_data = HDF5Dataset(train_data_filename, leg_swaps, arm_swaps, generate_heatmaps=generate_heatmaps)
val_data = HDF5Dataset(val_data_filename, leg_swaps, arm_swaps, generate_heatmaps=generate_heatmaps)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

model = network(loss_function, train_data.image_size, noise_length=noise_length).cuda()
if start_epoch > 0:
    state_dict = torch.load("output/{}/state_dict/network_{}.pth".format(output_folder, start_epoch - 1))
    model.load_state_dict(state_dict)
model.training = True
wandb.watch(model, log_freq=len(train_loader)//batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
for e in range(start_epoch, end_epoch):
    train_loss = 0
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        batch = next(train_iter)
        optimizer.zero_grad()

        if sample_method == "mixed":
            loss = model.mixed_sample_backward(batch, samples)
        elif sample_method == "select":
            losses = model.min_sample_loss(batch, samples)
            loss = torch.mean(losses)
            loss.backward()
        else:
            losses = model.loss(model(batch), batch)
            loss = torch.mean(losses)
            loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    with torch.no_grad():
        val_loss = 0
        val_iter = iter(val_loader)
        for i in range(len(val_loader)):
            batch = next(val_iter)
            optimizer.zero_grad()

            if sample_method == "mixed":
                loss = model.mixed_sample_loss(batch, samples)
            elif sample_method == "select":
                losses = model.min_sample_loss(batch, samples)
                loss = torch.mean(losses)
            else:
                losses = model.loss(model(batch), batch)
                loss = torch.mean(losses)
            val_loss += loss.item()
    
    wandb.log({
        "epoch": e,
        "training loss": train_loss / len(train_loader),
        "validation loss": val_loss / len(val_loader),
    })

    if (e+1) % checkpoint_freq == 0:
        torch.save(model.state_dict(), "output/{}/state_dict/network_{}.pth".format(output_folder, e))