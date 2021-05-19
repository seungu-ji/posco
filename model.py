"""
arxiv link: https://arxiv.org/abs/1703.10593
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='inorm'):
        super(CycleGAN, self).__init__()

        self.enc1 = CBR2d(in_channels, 1 * nker, kernel_size=7, padding=(7-1)//2, norm=norm, relu=0.0, stride=1)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)

        res = []
        for i in range(9):
            res += [ResBlock(4 * nker, 4 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=1)]

        self.res = nn.Sequential(*res)

        self.dec1 = DECBRD2d(4 * nker, 2 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)
        self.dec2 = DECBRD2d(2 * nker, 1 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)
        self.dec3 = CBR2d(1 * nker, out_channels, kernel_size=7, padding=(7-1)//2, norm=None, relu=None, stride=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        x = F.tanh(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(in_channels, 1 * nker, kernel_size=4, stride=2, padding=1, norm=None, relu=0.2, bias=False)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
                
        x = F.sigmoid(x)

        return x


class DECBRD2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0, drop=0.5):
        super().__init__()

        layers = []
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding, bias=bias)]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        if not drop is None:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.ReflectionPad2d(padding=padding)]
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=0, bias=bias)]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        # 1st conv
        layers += [CBR2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=relu)]

        # 2nd conv
        layers += [CBR2d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=None)]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)
