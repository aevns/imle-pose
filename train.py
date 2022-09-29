import argparse

import os
import random

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import models.utils.loss_functions as lf
from dataset import HDF5Dataset
from models.unet import UNet

# TODO: Use a config. for this instead of simply altering the training script, OR
# split this into two scripts (train, and one that operates like a config)
#########################################################################
print(torch.cuda.get_arch_list())
print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', '-d')
parser.add_argument('--batchsize', '-b', nargs='?', default=64, type=int)
parser.add_argument('--model', '-m', nargs='?', default='UNet')
parser.add_argument('--start', '-s', nargs='?', default=0, type=int)
parser.add_argument('--end', '-e', nargs='?', default=400, type=int)
parser.add_argument('--loss', '-l', nargs='?', default='gaussian')
parser.add_argument('--samples', '-sm', nargs='?', default=10, type=int)
parser.add_argument('--combine', '-c', nargs='?', default='select')
parser.add_argument('--armswaps', '-as', nargs='?', default=0, type=float)
parser.add_argument('--legswaps', '-ls', nargs='?', default=0, type=float)
parser.add_argument('--output', '-o', nargs='?', default='unnamed')
args = parser.parse_args()
torch.autograd.set_detect_anomaly(False)

start_epoch = args.start
num_epochs = args.end
batch_size = args.batchsize

annotation_file = args.dataroot + "person_keypoints_train.json"
data_file = args.dataroot + "train.hdf5"
leg_swaps = args.legswaps
arm_swaps = args.armswaps

#TODO: use model input for different models
network = UNet
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
model_description = network.name + "_" + str(noise_length) + "_" + sample_method + "_" + str(samples) + "_" + loss_function.__name__
#########################################################################

os.makedirs(os.path.dirname("output/{}/state_dict/".format(output_folder)), exist_ok=True)
os.makedirs(os.path.dirname("output/{}/training_log/".format(output_folder)), exist_ok=True)

train_data = HDF5Dataset(data_file, leg_swaps, arm_swaps, generate_heatmaps=generate_heatmaps)

train_loader = torch.utils.data.DataLoader(
    train_data,
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

loss_history = []
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
for e in range(start_epoch, num_epochs):
    logfile = open("output/{}/training_log/log.csv".format(output_folder), "w+")
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        batch = next(train_iter)
        optimizer.zero_grad()

        if sample_method == "mixed":
            losses = model.mixed_sample_loss(batch, samples)
        elif sample_method == "select":
            losses = model.min_sample_loss(batch, samples)
        else:
            losses = model.loss(model(batch), batch)
        loss = torch.mean(losses)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        if i%(len(train_loader)//10)==0:
            print("Epoch {}, iteration {} of {} ({} %), loss={}".format(e, i, len(train_loader), 100*i//len(train_loader), loss_history[-1]))
            logfile.write("Epoch:,{},iteration:,{},of,{},loss:,{}, model,{}\n".format(e, i, len(train_loader), loss_history[-1], model_description))
    torch.save(model.state_dict(), "output/{}/state_dict/network_{}.pth".format(output_folder, e))
    logfile.flush()
    logfile.close()