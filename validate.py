import argparse
import numpy as np
import torch
import os
import re

import matplotlib.pyplot as plt

import models.utils.loss_functions as lf
from dataset import HDF5Sampler
from dataset import HDF5Dataset
from models.unet import UNet
from models.unet_pretrained import UNetPretrained

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', '-d', nargs='?', default="data/stick/val.hdf5")
parser.add_argument('--batchsize', '-b', nargs='?', default=64, type=int)
parser.add_argument('--model', '-m', nargs='?', default='UNet', choices=['UNet', 'UNetPretrained'])
parser.add_argument('--folders', '-f', nargs='+', default=[
    "mse_1", "mse_m4", "mse_m8", "mse_m16", "mse_s4", "mse_s8", "mse_s16",
    "gauss_1", "gauss_m4", "gauss_m8", "gauss_m16", "gauss_s4", "gauss_s8", "gauss_s16",
    "dkl_1", "dkl_m4", "dkl_m8", "dkl_m16", "dkl_s4", "dkl_s8", "dkl_s16"])
parser.add_argument('--checkpoint', '-cp', nargs='?', default="network_899.pth")
parser.add_argument('--allcheckpoints', '-a', action='store_true')
parser.add_argument('--metric', '-mt', nargs='?', default='dkl', choices=['mse', 'gaussian', 'dkl'])
parser.add_argument('--samples', '-sm', nargs='?', default=100, type=int)
parser.add_argument('--combine', '-c', nargs='?', default='select', choices=['mixed', 'select'])
parser.add_argument('--legswaps', '-ls', nargs='?', default=0.5, type=float)
parser.add_argument('--armswaps', '-as', nargs='?', default=0.1, type=float)
parser.add_argument('--seed', nargs='?', default=145, type=int)
parser.add_argument('--save', nargs='?', default=None)
args = parser.parse_args()

#########################################################################

data_file = args.datafile
batch_size = args.batchsize
network = UNet if args.model == 'UNet' else UNetPretrained
leg_swaps = args.legswaps
arm_swaps = args.armswaps

metrics = {
    'dkl': {'gauss': lf.gaussian_dkl, 'dkl': lf.heatmap_target_dkl},
    'gaussian': {'gauss': lf.gaussian_nll},
    'mse': {'mse': lf.heatmap_mpjpe, 'gauss': lf.gaussian_mpjpe, 'dkl': lf.heatmap_mpjpe},
}
metric_labels = {'dkl': 'KL Divergence (nats)', 'gaussian': 'Negative Log-Likelihood (nats)', 'mse': 'MPJPE (px)'}

folder_names = []
loss_functions = []
sample_methods = []
for folder in args.folders:
    loss_type = folder.split('_')[0]
    if loss_type not in metrics[args.metric]:
        print("skipping {}: no {} metric for {} models".format(folder, args.metric, loss_type))
        continue
    folder_names.append(folder)
    loss_functions.append(metrics[args.metric][loss_type])
    sample_methods.append("constant" if folder.split('_')[-1] == '1' else args.combine)

#########################################################################

all_losses = []
all_epochs = []
for m in range(len(folder_names)):

    loss_history = []
    epochs = []

    dir = "output/{}/state_dict/".format(folder_names[m])
    if args.allcheckpoints:
        files = os.listdir(dir)
        files.sort(key = lambda x: int(re.search(r'\d+', x).group()))
    else:
        files = [args.checkpoint]
    for filename in files:

        val_data = HDF5Dataset(
            data_file,
            leg_swaps=leg_swaps,
            arm_swaps=arm_swaps,
            generate_heatmaps=True,
            device="cuda:0")

        val_sampler = HDF5Sampler(
            data_source=val_data,
            seed=args.seed)

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

        val_iter = iter(val_loader)
        val_loss = 0
        for i in range(len(val_loader)):
            batch = {k:v.cuda(0, non_blocking = True) for k, v in next(val_iter).items()}

            if sample_methods[m] == "mixed":
                loss = model.mixed_sample_loss(batch, args.samples) * batch['image'].shape[0]
            elif sample_methods[m] == "select":
                losses = model.min_sample_loss(batch, args.samples)
                loss = torch.sum(losses)
            elif sample_methods[m] == "constant":
                losses = model.unconditioned_loss(batch)
                loss = torch.sum(losses)
            val_loss += loss.item() / len(val_loader.dataset)
        print("{}, {}, {}, {}".format(folder_names[m], filename, val_loss, metric_labels[args.metric]))

        loss_history.append(val_loss)
        epochs.append(int(re.search(r'\d+', filename).group()))
        del val_iter
        del val_loader
    all_losses.append(loss_history)
    all_epochs.append(epochs)
    del loss_history

fig = plt.figure()
ax = fig.add_subplot()
#ax.set_yscale('log')
if (args.allcheckpoints):
    for i in range(len(all_losses)):
        ax.plot(all_epochs[i], all_losses[i], marker='o', label="{}".format(folder_names[i]))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_labels[args.metric])
    ax.legend()
if args.save:
    plt.savefig(args.save)
else:
    plt.show()
