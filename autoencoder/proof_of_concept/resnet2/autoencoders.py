import torch.nn as nn
from autoencoder.proof_of_concept.resnet2.full_reactivation import FullPreactivationResidualBlock
from autoencoder.proof_of_concept.resnet2.residual import ResidualBlock, ResidualBlockDownsample, ResidualBlockUpsample


class ResNet_v1(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            FullPreactivationResidualBlock(64, 128),
            FullPreactivationResidualBlock(128, 256),
            FullPreactivationResidualBlock(256, 512),
        )

        self.upsample = nn.Sequential(
            FullPreactivationResidualBlock(512, 256),
            FullPreactivationResidualBlock(256, 128),
            FullPreactivationResidualBlock(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x


class ResNet_v2(nn.Module):
    """performs well but slow. no overfits significally"""
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            FullPreactivationResidualBlock(64, 64),
            FullPreactivationResidualBlock(64, 64),
            FullPreactivationResidualBlock(64, 128),
            FullPreactivationResidualBlock(128, 128),
            FullPreactivationResidualBlock(128, 128),
            FullPreactivationResidualBlock(128, 256),
            FullPreactivationResidualBlock(256, 256),
        )

        self.upsample = nn.Sequential(
            FullPreactivationResidualBlock(256, 256),
            FullPreactivationResidualBlock(256, 128),
            FullPreactivationResidualBlock(128, 128),
            FullPreactivationResidualBlock(128, 128),
            FullPreactivationResidualBlock(128, 64),
            FullPreactivationResidualBlock(64, 64),
            FullPreactivationResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

class ResNet_v3(nn.Module):
    """overfits a lot after many epochs"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(24, 24),
            ResidualBlockDownsample(24, 48),
            ResidualBlock(48, 48),
            ResidualBlockDownsample(48, 96),
            ResidualBlock(96, 96),
            ResidualBlockDownsample(96, 192),
            ResidualBlock(192, 192),
        )

        self.upsample = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 96),
            ResidualBlock(96, 96),
            ResidualBlockUpsample(96, 48),
            ResidualBlock(48, 48),
            ResidualBlockUpsample(48, 24),
            ResidualBlock(24, 24),
            nn.Conv2d(24, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x
