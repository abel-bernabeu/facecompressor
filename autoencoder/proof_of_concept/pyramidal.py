import torch
import torch.nn as nn
import autoencoder.proof_of_concept.transition as transition
import autoencoder.proof_of_concept.resnet as resnet
import autoencoder.proof_of_concept.transition as transition


class Resnet6(nn.Module):

    def __init__(self, channels, kernel_size, activation, residual_block_factory):
        super().__init__()

        self.network = nn.Sequential(
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class Resnet6Factory():

    def __init__(self, residual_block_factory):
        super().__init__()
        self.residual_block_factory = residual_block_factory

    def get(self, channels, kernel_size, activation):
        return Resnet6(channels, kernel_size, activation, self.residual_block_factory)


class Resnet3(nn.Module):

    def __init__(self, channels, kernel_size, activation, residual_block_factory):
        super().__init__()

        self.network = nn.Sequential(
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
            residual_block_factory.get(channels, kernel_size, activation),
        )

    def forward(self, x):
        x = self.network(x)
        return x


class Resnet3Factory():

    def __init__(self, residual_block_factory):
        super().__init__()
        self.residual_block_factory = residual_block_factory

    def get(self, channels, kernel_size, activation):
        return Resnet3(channels, kernel_size, activation, self.residual_block_factory)


class Pyramidal5(nn.Module):

    def __init__(self,
        n,
        pyramidal_block_factory,
        pyramidal_last_block_factory,
        downscale_factory,
        upscale_factory,
        activation='relu',
    ):
        super().__init__()

        self.pre    = transition.Transition(3, n, 3, activation)
        self.post   = transition.Transition(n, 3, 3, activation, final=True)

        self.enc1   = pyramidal_block_factory.get(n * 1, 3, activation)
        self.enc2   = pyramidal_block_factory.get(n * 2, 3, activation)
        self.enc3   = pyramidal_block_factory.get(n * 4, 3, activation)
        self.enc4   = pyramidal_block_factory.get(n * 8, 3, activation)
        self.enc5   = pyramidal_last_block_factory.get(n * 16, 3, activation)

        self.dec5   = pyramidal_last_block_factory.get(n * 16, 3, activation)
        self.dec4   = pyramidal_block_factory.get(n * 8, 3, activation)
        self.dec3   = pyramidal_block_factory.get(n * 4, 3, activation)
        self.dec2   = pyramidal_block_factory.get(n * 2, 3, activation)
        self.dec1   = pyramidal_block_factory.get(n * 1, 3, activation)

        self.down2  = downscale_factory.get(n * 1, n * 2, activation)
        self.down3  = downscale_factory.get(n * 2, n * 4, activation)
        self.down4  = downscale_factory.get(n * 4, n * 8, activation)
        self.down5  = downscale_factory.get(n * 8, n * 16, activation)

        self.up5    = upscale_factory.get(n * 16, n * 8, activation)
        self.up4    = upscale_factory.get(n * 8, n * 4, activation)
        self.up3    = upscale_factory.get(n * 4, n * 2, activation)
        self.up2    = upscale_factory.get(n * 2, n * 1, activation)

    def forward(self, x):
        x1 = x

        x1 = self.pre(x1)
        x1 = self.enc1(x1)

        x2 = x1
        x2 = self.down2(x2)
        x2 = self.enc2(x2)

        x3 = x2
        x3 = self.down3(x3)
        x3 = self.enc3(x3)

        x4 = x3
        x4 = self.down4(x4)
        x4 = self.enc4(x4)

        x5 = x4
        x5 = self.down5(x5)
        x5 = self.enc5(x5)

        y5 = x5

        y4 = y5
        y4 = self.dec5(y4)
        y4 = self.up5(y4)
        y4 = y4 + x4

        y3 = y4
        y3 = self.dec4(y3)
        y3 = self.up4(y3)
        y3 = y3 + x3

        y2 = y3
        y2 = self.dec3(y2)
        y2 = self.up3(y2)
        y2 = y2 + x2

        y1 = y2
        y1 = self.dec2(y1)
        y1 = self.up2(y1)
        y1 = y1 + x1

        y = y1
        y = self.dec1(y)
        y = self.post(y)

        return y


def get_default():
    return Pyramidal5(
        48,
        Resnet6Factory(resnet.FullPreactivationResidualBlockFactory()),
        Resnet6Factory(resnet.FullPreactivationResidualBlockFactory()),
        transition.DownscaleConv2dFactory(),
        transition.UpscaleConvTranspose2dFactory()
    )