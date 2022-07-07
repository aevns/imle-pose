
import torch
import torchvision.transforms as transforms
import h5py

FloatTensor =  torch.cuda.FloatTensor
Device = "cuda:0"

data_file = "./data/stick/val.hdf5"
with h5py.File(data_file, 'r') as df:
    poses = torch.from_numpy(df['poses'][...]).to(Device)
    images  = torch.from_numpy(df['images'][...]).to(Device)


mean = torch.mean(images,dim=(0,2,3))
std = torch.std(images,dim=(0,2,3))

print(mean)
print(std)