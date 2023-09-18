"""
Modified from https://github.com/pytorch/vision.git
"""
import math

import torch.nn as nn


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(self, features, num_share_layers=None, dropout=0.5):
        super(VGG, self).__init__()
        conv_layers = features
        conv_layers.append(nn.Flatten())
        clf_layers = nn.ModuleList([
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)]
        )
        all_layers = []
        all_layers.extend(conv_layers)
        all_layers.extend(clf_layers)

        # print(all_layers)
        share_depth_to_cut = {
            0: 0,

            1: 3,

            2: 6,

            3: 8,

            4: 11,

            5: 13,

            6: 16,

            7: 18,

            8: 21,

            9: 26,

            10: 28,

            11: 29
        }
        # re-arrange conv and clf
        cut_layer = share_depth_to_cut[num_share_layers]
        self.conv = nn.Sequential(*all_layers[:cut_layer])
        self.clf = nn.Sequential(*all_layers[cut_layer:])

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        if len(self.conv):
            x = self.conv(x)
        if len(self.clf):
            x = self.clf(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # return nn.Sequential(*layers)
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(num_share_layers=8, dropout=0.5):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_share_layers=num_share_layers, dropout=dropout)