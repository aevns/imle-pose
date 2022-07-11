import torch
import torch.nn as nn
import models.utils.loss_functions as lf

class DCIGNet(nn.Module):
    def __init__(self, loss_function = lf.heatmap_target_mse):
        super(DCIGNet, self).__init__()

        self.loss = loss_function

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # maxpool

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # maxpool

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # maxpool

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(128)
        # maxpool

        self.vectorize = nn.Linear(128 * 4 * 3, 248)
        self.devectorize =  nn.Linear(256, 128 * 4 * 3)
    
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6_1 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6_2 = nn.BatchNorm2d(32)
        
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7_1 = nn.BatchNorm2d(32)
        self.deconv7 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7_2 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(32, 17, kernel_size=1, stride=1, padding=0)

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
        out = self.relu(self.bn1(self.conv1(x['image'])))
        out = self.maxpool(out)

        out = self.relu(self.bn2(self.conv2(out)))
        out = self.maxpool(out)

        out = self.relu(self.bn3(self.conv3(out)))
        out = self.maxpool(out)

        out = self.relu(self.bn4_1(self.conv4(out)))
        out = self.maxpool(out)
        out = self.relu(self.vectorize(out.view(-1, 128 * 4 * 3)))
        out = self.relu(self.devectorize(torch.cat((out, z), dim=-1))).view(-1, 128, 4, 3)
        out = self.relu(self.bn4_2(self.deconv4(out)))
        
        out = self.relu(self.bn5_1(self.conv5(out)))
        out = self.relu(self.bn5_2(self.deconv5(out)))
        
        out = self.relu(self.bn6_1(self.conv6(out)))
        out = self.relu(self.bn6_2(self.deconv6(out)))
        
        out = self.relu(self.bn7_1(self.conv7(out)))
        out = self.relu(self.bn7_2(self.deconv7(out)))

        return self.conv8(out)
    
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