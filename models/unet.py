from math import log
from math import floor
import torch
import torch.nn as nn

class UNetCenter(nn.Module):
    def __init__(self, input_shape, noise_length):
        super(UNetCenter, self).__init__()
        self.input_shape = input_shape
        self.noise_length = noise_length
        self.in_channels = input_shape[0]
        self.out_channels = input_shape[0]

        self.relu = nn.ReLU(inplace=True)
        self.vectorize = nn.Linear(torch.prod(self.input_shape).item(), 128 - noise_length)
        self.devectorize =  nn.Linear(128, torch.prod(self.input_shape).item())
    
    def _initialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
    
    def forward(self, x, z):
        out = self.relu(self.vectorize(x.view(-1, torch.prod(self.input_shape))))
        out = torch.cat((out, z), dim=-1)
        out = self.relu(self.devectorize(out)).view(-1, *self.input_shape)
        return out

class UNetStage(nn.Module):
    def __init__(self, center, in_channels, out_channels):
        super(UNetStage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, center.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(center.in_channels)
        self.center = center
        self.deconv2 = nn.ConvTranspose2d(center.out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def _initialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
    
    def forward(self, x, z):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.center(out, z)
        out = self.relu(self.bn2(self.deconv2(out)))
        out = torch.cat((out, x), dim=1)
        out = self.relu(self.bn3(self.conv3(out)))
        return out

class UNet(nn.Module):
    name = "unet"
    implicit = True
    min_channels = 32
    max_channels = 128

    def __init__(self, loss_function, input_size, noise_length = 8, n_stages = None):
        super(UNet, self).__init__()

        self.loss = loss_function
        self.input_size = input_size
        self.noise_length = noise_length
        if n_stages == None:
            self.n_stages = torch.floor(torch.log2(torch.min(input_size))).to(torch.int32).item() - 1
        else:
            self.n_stages = n_stages

        center_shape = torch.tensor([
            UNet.max_channels,
            torch.div(self.input_size[0],(2**self.n_stages), rounding_mode='floor'),
            torch.div(self.input_size[1],(2**self.n_stages), rounding_mode='floor')
            ]).to(torch.int32)
        self.net = UNetCenter(center_shape, noise_length)

        for i in range(1, self.n_stages + 1):
            channels = floor(UNet.max_channels**(1 - i/self.n_stages)*UNet.min_channels**(i/self.n_stages))
            self.net = UNetStage(self.net, channels, channels)
        
        self.initial = nn.Conv2d(3, UNet.min_channels, kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(UNet.min_channels, 17, kernel_size=1, stride=1, padding=0)

    def _initialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x, z):
        out = self.initial(x['image'])
        out = self.net(out, z)
        out = self.final(out)
        return {'pose': UNet.gaussian_fit(UNet.normalize(out)), 'heatmap': out}
    
    def get_sample(self, x):
        z = torch.randn((x['image'].shape[0], self.noise_length), device = x['image'].device)
        return self.forward(x, z)

    def unconditioned_loss(self, x):
        z = torch.zeros((x['image'].shape[0], self.noise_length), device = x['image'].device)
        return self.loss(self.forward(x, z), x)

    def min_sample_loss(self, x, n):
        with torch.no_grad():
            for s in range(n):
                z = torch.randn((x['image'].shape[0], self.noise_length), device = x['image'].device)
                pred = self.forward(x, z)
                if s == 0:
                    noise = z
                    losses = self.loss(pred, x)
                else:
                    l = self.loss(pred, x)
                    mask = l < losses
                    losses[mask] = l[mask]
                    noise[mask] = z[mask]
        return self.loss(self.forward(x, noise), x)

    def mixed_sample_backward(self, x, n):
        net_grad = {}
        for i in range(n):
            z = torch.randn((x['image'].shape[0], self.noise_length), device = x['image'].device)
            pred = self.forward(x, z)
            nll = torch.mean(self.loss(pred, x))
            self.zero_grad()
            nll.backward()
            with torch.no_grad():
                if i==0:
                    for name, param in self.named_parameters():
                        net_grad[name] = param.grad.detach().clone()
                    net_nll = nll
                else:
                    stabilizer = torch.max(net_nll, nll)
                    for name, param in self.named_parameters():
                        net_grad[name] = (
                            (torch.exp(net_nll - stabilizer) * param.grad
                            + torch.exp(nll - stabilizer) * net_grad[name])
                            / (torch.exp(net_nll - stabilizer) + torch.exp(nll - stabilizer)))
                    net_nll = -torch.log(torch.exp(-net_nll + stabilizer) + torch.exp(-nll + stabilizer)) + stabilizer
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.grad = net_grad[name]
            return net_nll

    def mixed_sample_loss(self, x, n):
        with torch.no_grad():  
            for i in range(n):
                z = torch.randn((x['image'].shape[0], self.noise_length), device = x['image'].device)
                pred = self.forward(x, z)
                nll = torch.mean(self.loss(pred, x))
                if i==0:
                    net_nll = nll
                else:
                    stabilizer = torch.max(net_nll, nll)
                    net_nll = -torch.log(torch.exp(-net_nll + stabilizer) + torch.exp(-nll + stabilizer)) + stabilizer
            return net_nll

    def normalize(pred):
        n, c, h, w = pred.shape

        max_ = torch.max(torch.max(pred, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
        z = torch.sum(torch.exp(pred - max_), (2, 3)).view(n, c, 1, 1)
        h_norm = torch.exp(pred - max_) / z

        return h_norm
     
    def gaussian_fit(pred):
        n, c, h, w = pred.shape

        x_vals = torch.linspace(0, w-1, w, device=pred.device).unsqueeze(0)
        y_vals = torch.linspace(0, h-1, h, device=pred.device).unsqueeze(1)

        x_means = torch.sum(pred * x_vals, dim = (2, 3))
        y_means = torch.sum(pred * y_vals, dim = (2, 3))
        
        xn = (x_vals - x_means.view(n, c, 1, 1))
        yn = (y_vals - y_means.view(n, c, 1, 1))

        x_var = 1/12 + torch.sum(pred * xn * xn, dim=(2,3))
        y_var = 1/12 + torch.sum(pred * yn * yn, dim=(2,3))
        xy_covar = torch.sum(pred * xn * yn, dim=(2,3))

        return torch.stack((x_means, y_means, x_var, y_var, xy_covar), -1)