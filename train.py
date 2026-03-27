import argparse

import os
from math import prod
import random

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import models.utils.loss_functions as lf
from dataset import HDF5Sampler
from dataset import HDF5Dataset
from models.unet import UNet
from models.unet import UNetLarge
from models.unet_pretrained import UNetPretrained

#import wandb

#wandb.login(key='9a92298caf7b15ab1719f839763164b8932817a9')

print(torch.cuda.get_arch_list())
print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', '-d')
parser.add_argument('--learningrate', '-lr', nargs='?', default=0.001, type=float)
parser.add_argument('--weightdecay', '-wd', nargs='?', default=0.01, type=float)
parser.add_argument('--batchsize', '-b', nargs='?', default=64, type=int)
parser.add_argument('--model', '-m', nargs='?', default='UNet')
parser.add_argument('--start', '-s', nargs='?', default=0, type=int)
parser.add_argument('--checkpoints', '-cp', nargs='?', default=1, type=int)
parser.add_argument('--end', '-e', nargs='?', default=400, type=int)
parser.add_argument('--loss', '-l', nargs='?', default='gaussian')
parser.add_argument('--samples', '-sm', nargs='?', default=30, type=int)
parser.add_argument('--combine', '-c', nargs='?', default='select')
parser.add_argument('--armswaps', '-as', nargs='?', default=0, type=float)
parser.add_argument('--legswaps', '-ls', nargs='?', default=0, type=float)
parser.add_argument('--output', '-o', nargs='?', default='unnamed')
args = parser.parse_args()
print(args)
torch.autograd.set_detect_anomaly(False)

start_epoch = args.start
end_epoch = args.end
checkpoint_freq = (end_epoch - start_epoch) / args.checkpoints
learning_rate = args.learningrate
weight_decay = args.weightdecay
batch_size = args.batchsize

train_data_filename = args.dataroot + "train.hdf5"
val_data_filename = args.dataroot + "val.hdf5"

leg_swaps = args.legswaps
arm_swaps = args.armswaps

#wandb.init(
#    project="imle-pose",
#   config = {'args': args})

if args.model == 'UNet':
    network = UNet
elif args.model == 'UNetPretrained':
    network = UNetPretrained
elif args.model == 'UNetLarge':
    network = UNetLarge
noise_length = 8

samples = args.samples
if args.loss == 'mse':
    train_loss_fn = lf.heatmap_target_mse
    val_loss_fn = lf.heatmap_target_mse
    generate_heatmaps = True
elif args.loss == 'gaussian':
    train_loss_fn = lambda pred, x : lf.gaussian_nll(pred, x) + 0.001 * lf.label_loss(pred, x)
    val_loss_fn = lf.gaussian_nll
    generate_heatmaps = False
elif args.loss == 'dkl':
    train_loss_fn = lambda pred, x : lf.heatmap_target_dkl(pred, x) + 0.001 * lf.label_loss(pred, x)
    val_loss_fn = lf.heatmap_target_dkl
    generate_heatmaps = True

sample_method = args.combine

output_folder = args.output

os.makedirs(os.path.dirname("{}/state_dict/".format(output_folder)), exist_ok=True)
os.makedirs(os.path.dirname("{}/training_log/".format(output_folder)), exist_ok=True)

train_data = HDF5Dataset(
    train_data_filename,
    leg_swaps,
    arm_swaps,
    generate_heatmaps=generate_heatmaps,
    device="cuda:0")

val_data = HDF5Dataset(
    val_data_filename,
    leg_swaps,
    arm_swaps,
    generate_heatmaps=generate_heatmaps,
    device="cuda:0")

train_sampler = HDF5Sampler(
    data_source=train_data)

val_sampler = HDF5Sampler(
    data_source=val_data)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=0,
    sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = batch_size,
    num_workers=0,
    sampler=val_sampler)

model = network(train_loss_fn, train_data.image_size, noise_length=noise_length).cuda(0)

if start_epoch > 0:
    state_dict = torch.load("{}/state_dict/network_{}.pth".format(output_folder, start_epoch - 1))
    model.load_state_dict(state_dict)
#wandb.watch(model, log_freq=len(train_loader)//batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
for e in range(start_epoch, end_epoch):

    # TRAINING
    model.training = True
    model.loss = train_loss_fn
    train_loss = 0
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        batch = {k:v.cuda(0, non_blocking = True) for k, v in next(train_iter).items()}
        optimizer.zero_grad()

        if sample_method == "mixed":
            loss = model.mixed_sample_backward(batch, samples)
        elif sample_method == "select":
            losses = model.min_sample_loss(batch, samples)
            loss = torch.mean(losses)
            loss.backward()
        elif sample_method == "constant":
            losses = model.unconditioned_loss(batch)
            loss = torch.mean(losses)
            loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # VALIDAITON
    model.training = False
    model.loss = val_loss_fn
    with torch.no_grad():
        val_loss = 0
        val_iter = iter(val_loader)
        for i in range(len(val_loader)):
            batch = {k:v.cuda(0, non_blocking = True) for k, v in next(val_iter).items()}
            optimizer.zero_grad()

            if sample_method == "mixed":
                loss = model.mixed_sample_loss(batch, samples)
            elif sample_method == "select":
                losses = model.min_sample_loss(batch, samples)
                loss = torch.mean(losses)
            elif sample_method == "constant":
                losses = model.unconditioned_loss(batch)
                loss = torch.mean(losses)
            val_loss += loss.item()
    
    #LOGGING
    #wandb.log({
    #    "epoch": e,
    #    "training loss": train_loss / len(train_loader),
    #    "validation loss": val_loss / len(val_loader),
    #})
    print("epoch:", e, "training loss:", train_loss / len(train_loader), "validation loss:", val_loss / len(val_loader))
    if (e+1) % checkpoint_freq == 0:
        torch.save(
            model.state_dict(),
            "{}/state_dict/network_{}.pth".format(output_folder, e))