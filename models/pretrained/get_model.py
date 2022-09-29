import torch

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
torch.save(resnet, "resnet50.pt")