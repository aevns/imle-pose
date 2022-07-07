import torch.nn as nn
from .layers.resnet import BasicBlock

class Basic(nn.Module):
    def __init__(self):
        super(Basic, self).__init__()

        self.deconv_dim = (64,64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)

        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.final_layer = nn.Conv2d(32, 17, kernel_size=1, stride=1, padding=0)

    def _initialize(self):
        for name, m in self.named_modules():
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

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x['image']))))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.deconv1(out)))
        out = self.relu(self.bn4(self.deconv2(out)))
        out = self.final_layer(out)
        return out