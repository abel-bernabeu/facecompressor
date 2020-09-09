import torch.nn as nn


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'leaky_relu':
        return nn.LeakyReLU()
    raise ValueError(f'activation ot supported: {activation}')


class BasicBlock2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation):
        super().__init__()

        padding = (kernel_size -1) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
        )

    def forward(self, x):
        return self.network(x)


class BasicTransposeBlock2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, final=False):
        super().__init__()

        padding = (kernel_size - 1) // 2
        if not final:
            self.network = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=(0 if stride == 1 else 1)),
                nn.BatchNorm2d(out_channels),
                get_activation(activation),
            )
        else:
            self.network = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=(0 if stride == 1 else 1)),
                nn.Hardtanh(0., 1.)
            )

    def forward(self, x):
        return self.network(x)


class Reshape(nn.Module):

    def __init__(self, channels, height, width):
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.reshape(-1, self.channels, self.height, self.width)
