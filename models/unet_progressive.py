import torch
import torch.nn as nn
import models.utils.loss_functions as lf

class UNetProgressive(nn.Module):
    def __init__(self, loss_function = lf.heatmap_target_mse):
        super(UNetProgressive, self).__init__()

        self.loss = loss_function

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(8, 32, bias=False)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # maxpool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # maxpool
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)

        # maxpool
        self.linear4 = nn.Linear(8, 128, bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(64)

        # concat(4,3)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(64)

        # concat(5,2)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6_1 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6_2 = nn.BatchNorm2d(32)
        
        # concat(6,1)
        self.conv7_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv7_2 = nn.Conv2d(32, 17, kernel_size=1, stride=1, padding=0)

    def _initialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

    def forward(self, x, z):
        block1 = self.conv1(x['image']) + self.linear1(z).view(-1, 32, 1, 1)
        block1 = self.relu(self.bn1(block1))

        block2 = self.maxpool(block1)
        block2 = self.relu(self.bn2(self.conv2(block2)))
        
        block3 = self.maxpool(block2)
        block3 = self.relu(self.bn3_1(self.conv3_1(block3)))
        block3 = self.relu(self.bn3_2(self.conv3_2(block3)))
        
        out = self.maxpool(block3)
        out = self.conv4(out) + self.linear4(z).view(-1, 128, 1, 1)
        out = self.relu(self.bn4_1(out))
        out = self.relu(self.bn4_2(self.deconv4(out)))
        
        out = torch.cat((out, block3), dim=1)
        out = self.relu(self.bn5_1(self.conv5(out)))
        out = self.relu(self.bn5_2(self.deconv5(out)))
        
        out = torch.cat((out, block2), dim=1)
        out = self.relu(self.bn6_1(self.conv6(out)))
        out = self.relu(self.bn6_2(self.deconv6(out)))
        
        out = torch.cat((out, block1), dim=1)
        out = self.relu(self.bn7(self.conv7_1(out)))
        out = (self.conv7_2(out))

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