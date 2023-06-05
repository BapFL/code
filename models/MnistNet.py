import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(4 * 4 * 50, 500),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.clf(x)
        return x