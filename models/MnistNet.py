import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self, num_share_layers=None):
        super(MnistNet, self).__init__()
        conv_layers = nn.ModuleList([
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()]
        )
        clf_layers = nn.ModuleList([
            nn.Linear(4 * 4 * 50, 500),
            nn.Linear(500, 10)]
        )
        all_layers = []
        all_layers.extend(conv_layers)
        all_layers.extend(clf_layers)

        share_depth_to_cut = {
            0: 0,

            1: 3,

            2: 7,

            3: 8,

            4: 9

        }

        cut_layer = share_depth_to_cut[num_share_layers]
        self.conv = nn.Sequential(*all_layers[:cut_layer])
        self.clf = nn.Sequential(*all_layers[cut_layer:])

    def forward(self, x):
        if len(self.conv):
            x = self.conv(x)
        if len(self.clf):
            x = self.clf(x)
        return x
