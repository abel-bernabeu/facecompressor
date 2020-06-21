import torch.nn as nn
import autoencoder.proof_of_concept.basic as basic


class Transition(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation, final=False):
        super().__init__()

        padding = (kernel_size - 1) // 2
        if not final:
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
                nn.BatchNorm2d(out_channels),
                basic.get_activation(activation),
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
                nn.Hardtanh(0., 1.)
            )

    def forward(self, x):
        return self.network(x)


class DownscaleConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2, 0),
            nn.BatchNorm2d(out_channels),
            basic.get_activation(activation),
        )

    def forward(self, x):
        return self.network(x)


class DownscaleConv2dFactory():

    def __init__(self):
        super().__init__()

    def get(self, in_channels, out_channels, activation):
        return DownscaleConv2d(in_channels, out_channels, activation)


class DownscaleMaxPool2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        if in_channels == out_channels:
            self.network = nn.MaxPool2d(2)
        else:
            self.network = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels),
                basic.get_activation(activation),
            )

    def forward(self, x):
        return self.network(x)


class DownscaleMaxPool2dFactory():

    def __init__(self):
        super().__init__()

    def get(self, in_channels, out_channels, activation):
        return DownscaleMaxPool2d(in_channels, out_channels, activation)


class DownscaleAvgPool2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        if in_channels == out_channels:
            self.network = nn.MaxPool2d(2)
        else:
            self.network = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels),
                basic.get_activation(activation),
            )

    def forward(self, x):
        return self.network(x)


class DownscaleAvgPool2dFactory():

    def __init__(self):
        super().__init__()

    def get(self, in_channels, out_channels, activation):
        return DownscaleAvgPool2d(in_channels, out_channels, activation)


class UpscaleConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
            nn.BatchNorm2d(out_channels),
            basic.get_activation(activation),
        )

    def forward(self, x):
        return self.network(x)


class UpscaleConvTranspose2dFactory():

    def __init__(self):
        super().__init__()

    def get(self, in_channels, out_channels, activation):
        return UpscaleConvTranspose2d(in_channels, out_channels, activation)


class UpscalePixelShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, 1),
            nn.BatchNorm2d(out_channels * 4),
            basic.get_activation(activation),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.network(x)


class UpscalePixelShuffleFactory():

    def __init__(self):
        super().__init__()

    def get(self, in_channels, out_channels, activation):
        return UpscaleConvTranspose2d(in_channels, out_channels, activation)