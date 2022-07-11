import torch.nn as nn
from .layers.resnet import ResNet
import models.utils.loss_functions as lf

class SimplePose(nn.Module):
    def __init__(self, loss_function = lf.heatmap_target_mse, norm_layer=nn.BatchNorm2d):
        super(SimplePose, self).__init__()
        
        self.loss = loss_function

        self.deconv_dim = (64,64,64)
        self._norm_layer = norm_layer
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(
            64, 17, kernel_size=1, stride=1, padding=0)
        self.preact = ResNet(f"resnet{18}")

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            512, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
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
        out = self.preact(x['image'])
        out = self.deconv_layers(out)
        out = self.final_layer(out)
        return out
    
    def sample(self, x):
        return self.forward(x), None