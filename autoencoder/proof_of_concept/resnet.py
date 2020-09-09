import torch.nn as nn
import autoencoder.proof_of_concept.basic as basic


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, activation):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            basic.get_activation(activation),
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
        )
        self.activation = basic.get_activation(activation)

    def forward(self, x):
        x = x + self.branch(x)
        x = self.activation(x)
        return x


class ResidualBlockFactory():

    def __init__(self):
        super().__init__()

    def get(self, channels, kernel_size, activation):
        return ResidualBlock(channels, kernel_size, activation)


class FullPreactivationResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, activation):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.branch = nn.Sequential(
            nn.BatchNorm2d(channels),
            basic.get_activation(activation),
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            basic.get_activation(activation),
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
        )

    def forward(self, x):
        x = x + self.branch(x)
        return x


class FullPreactivationResidualBlockFactory():

    def __init__(self):
        super().__init__()

    def get(self, channels, kernel_size, activation):
        return FullPreactivationResidualBlock(channels, kernel_size, activation)