import torch
import torch.nn as nn

class BasicProgressive(nn.Module):
    def __init__(self):
        super(BasicProgressive, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.linear1 = nn.Linear(4, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.linear2 = nn.Linear(4, 64)

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

    def forward(self, x, z): # (3, 64, 48) & (40,)
        stage1 = self.relu(self.bn1(self.conv1(x['image']))) # (32, 32, 24)
        stage1 = self.maxpool(stage1) # (32, 16, 12)
        out = self.conv2(stage1) + self.linear1(z).view(-1, 64, 1, 1)  # (64, 8, 6)
        out = self.relu(self.bn2(out))  # (64, 8, 6)
        out = self.relu(self.bn3(self.deconv1(out))) # (32, 16, 12)
        out = torch.cat((out, stage1), dim=1) # (64, 16, 12)
        out = self.relu(self.bn4(self.deconv2(out))) + self.linear2(z).view(-1, 64, 1, 1) # (64, 32, 24)
        out = self.relu(self.bn5(self.deconv3(out))) # (32, 64, 48)
        out = self.final_layer(out)# (17, 32, 24)
        return out
    
    def sample(self, x):
        z = torch.randn((x['image'].shape[0], 4))
        return self.forward(x, z), z
    
    def train_sample(self, x, n):
        for s in range(n):
            z = torch.randn((x['image'].shape[0], 4), device="cuda:0")
            pred = self.forward(x, z)
            if s == 0:
                noise = z
                losses = 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, x['target']), dim=(-3,-2,-1))
            else:
                l = 0.5 * torch.mean(nn.MSELoss(reduction='none')(pred, x['target']), dim=(-3,-2,-1))
                mask = l < losses
                losses[mask] = l[mask]
                noise[mask] = z[mask]

        return self.forward(x, noise), noise
