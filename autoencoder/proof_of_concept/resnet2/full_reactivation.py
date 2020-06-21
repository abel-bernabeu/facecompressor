import torch.nn as nn


class FullPreactivationResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2  = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels == out_channels:
            self._forward = self.forward1
        else:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            self._forward = self.forward2

    def forward(self, x):
        return self._forward(x)

    def forward1(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        return x + y

    def forward2(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        x = self.conv3(x)
        return x + y
