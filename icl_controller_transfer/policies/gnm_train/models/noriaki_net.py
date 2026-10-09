import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoriakiNet(nn.Module):
    def __init__(self, in_chs=3):
        super(NoriakiNet, self).__init__()
        kwargs = {"bn": True, "sample": CBR.DOWN, "activation": F.leaky_relu, "dropout": False}
        self.noriaki_net = nn.Sequential(
            nn.Conv2d(in_chs, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            CBR(64, 128, **kwargs),
            CBR(128, 256, **kwargs),
            # CBR(256, 512, **kwargs),
            # CBR(512, 512, **kwargs),
            # CBR(512, 512, **kwargs),
            # CBR(512, 512, **kwargs),
            CBR(256, 512, flatten=True, **kwargs),
        )

    def forward(self, x):
        return self.noriaki_net(x)

# test

class CBR(nn.Module):
    """Convolution-Batchnorm-(Dropout)-ReLU"""
    DOWN = True
    UP = False

    def __init__(self, in_channels, out_channels, flatten=False, bn=True, sample=DOWN,
                 activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        super(CBR, self).__init__()
        conv = nn.Conv2d if sample == self.DOWN else nn.ConvTranspose2d
        self._conv = conv(in_channels, out_channels, 4, stride=2, padding=1)
        self._bn = nn.BatchNorm2d(out_channels)
        self.flatten = flatten

    def forward(self, x):
        h = self._conv(x)
        if self.bn:
            h = self._bn(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        # average pooling
        h = F.avg_pool2d(h, 2)
        # squeeze to [B, C]
        if self.flatten:
            h = h.view(h.size(0), -1)
        return h
    



# kwargs = {"bn": True, "sample": CBR.DOWN, "activation": F.leaky_relu, "dropout": False}
# noriaki_net = nn.Sequential(
#     nn.Conv2d(in_chs, 64, 3, stride=1, padding=1),
#     nn.LeakyReLU(negative_slope=0.2),
#     CBR(64, 128, **kwargs),
#     CBR(128, 256, **kwargs),
#     CBR(256, 512, **kwargs),
#     CBR(512, 512, **kwargs),
#     CBR(512, 512, **kwargs),
#     CBR(512, 512, **kwargs),
#     CBR(512, 512, **kwargs),
# )