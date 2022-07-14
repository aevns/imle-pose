import torch
import torch.nn as nn

class VectorNet(nn.Module):
    name = "unet_vector"
    implicit = True
    def __init__(self, loss_function):
        super(VectorNet, self).__init__()

        self.loss = loss_function

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.vectorize = nn.Linear(256 * 4 * 3, 248)

        self.l1 = nn.Linear(256,17*20)
        self.l2 = nn.Linear(17*20,17*10)
        self.l3 = nn.Linear(17*10,17*5)
        self.c3 = nn.Conv1d(17, 17, 1)
        
    def _initialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    def forward(self, x, z):
        out = self.relu(self.bn1(self.conv1(x['image'])))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.vectorize(out.view(-1, 256 * 4 * 3)))
        out = torch.cat((out, z), dim=-1)
        out = self.relu(self.l1(out))
        out = self.relu(self.l2(out))
        out = self.relu(self.l3(out))
        temp = self.c3(out.view(-1, 17, 5))
        out = torch.cat((
            torch.abs(temp[:,:,0:2]),
            temp[:,:,2:4]**2,
            (torch.abs(temp[:,:,2]) * torch.abs(temp[:,:,3]) * (2 * torch.sigmoid(temp[:,:,4]) - 1)).unsqueeze(-1)
        ), dim=2)
        return out
    
    def sample(self, x):
        z = torch.randn((x['image'].shape[0], 8), device="cuda:0")
        return self.forward(x, z), z
    
    def train_sample(self, x, n):
        for s in range(n):
            z = torch.randn((x['image'].shape[0], 8), device="cuda:0")
            pred = self.forward(x, z)
            if s == 0:
                noise = z
                losses = self.loss(pred, x)
            else:
                l = self.loss(pred, x)
                mask = l < losses
                losses[mask] = l[mask]
                noise[mask] = z[mask]

        return self.forward(x, noise), noise