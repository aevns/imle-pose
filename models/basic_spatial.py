import torch
import torch.nn as nn
from models.utils.perlin_noise import perlin_ms

class BasicSpatial(nn.Module):
    def __init__(self, loss_function):
        super(BasicSpatial, self).__init__()

        self.loss = loss_function

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.final_layer = nn.Conv2d(32, 17, kernel_size=1, stride=1, padding=0)

    def _initialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, z): # (3, 64, 48), (1, 64, 48)
        block1 = torch.cat((x['image'], z), dim=1) # (4, 64, 48)
        block1 = self.relu(self.bn1(self.conv1(block1))) # (32, 32, 24)
        block1 = self.maxpool(block1) # (32, 16, 12)
        out = self.relu(self.bn2(self.conv2(block1))) # (64, 8, 6)
        out = self.relu(self.bn3(self.deconv1(out))) # (32, 16, 12)
        out = torch.cat((out, block1), dim=1) # (64, 16, 12)
        out = self.relu(self.bn4(self.deconv2(out))) # (32, 32, 24)
        out = self.relu(self.bn5(self.deconv3(out))) # (32, 64, 48)
        out = self.final_layer(out) # (17, 32, 24)
        return out
    
    def sample(self, x):
        z = perlin_ms([1, .5, .25, .125], x['image'].shape[0], 4, 3, device="cuda:0").unsqueeze(1)
        return self.forward(x, z), z
    
    def train_sample(self, x, n):
        for s in range(n):
            z = perlin_ms([1, .5, .25, .125], x['image'].shape[0], 4, 3, device="cuda:0").unsqueeze(1)
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