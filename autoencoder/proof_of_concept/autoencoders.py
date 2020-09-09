import autoencoder.proof_of_concept.basic as basic
import autoencoder.proof_of_concept.resnet2 as resnet
import autoencoder.proof_of_concept.transition as transition
import torch.nn as nn


class Basic(nn.Module):

    def __init__(self, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            basic.BasicBlock2d(3, 24, 3, 1, activation),
            basic.BasicBlock2d(24, 48, 3, 2, activation),
            basic.BasicBlock2d(48, 96, 3, 2, activation),
            basic.BasicBlock2d(96, 192, 3, 2, activation),
        )

        self.decoder = nn.Sequential(
            basic.BasicTransposeBlock2d(192, 96, 3, 2, activation),
            basic.BasicTransposeBlock2d(96, 48, 3, 2, activation),
            basic.BasicTransposeBlock2d(48, 24, 3, 2, activation),
            basic.BasicTransposeBlock2d(24, 3, 3, 1, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BasicFC(nn.Module):

    def __init__(self, width, height, hidden, activation='relu'):
        super().__init__()

        units1 = width // 32 * height // 32 * 512
        units2 = int(units1 * hidden)

        self.encoder = nn.Sequential(
            basic.BasicBlock2d(3, 16, 3, 1, activation),
            basic.BasicBlock2d(16, 32, 3, 2, activation),
            basic.BasicBlock2d(32, 64, 3, 2, activation),
            basic.BasicBlock2d(64, 128, 3, 2, activation),
            basic.BasicBlock2d(128, 256, 3, 2, activation),
            basic.BasicBlock2d(256, 512, 3, 2, activation),
            nn.Flatten(),
            nn.Linear(units1, units2),
            nn.Dropout(0.25),
        )

        self.decoder = nn.Sequential(
            nn.Linear(units2, units1),
            nn.Dropout(0.25),
            basic.Reshape(512, height // 32, width // 32),
            basic.BasicTransposeBlock2d(512, 256, 3, 2, activation),
            basic.BasicTransposeBlock2d(256, 128, 3, 2, activation),
            basic.BasicTransposeBlock2d(128, 64, 3, 2, activation),
            basic.BasicTransposeBlock2d(64, 32, 3, 2, activation),
            basic.BasicTransposeBlock2d(32, 16, 3, 2, activation),
            basic.BasicTransposeBlock2d(16, 3, 3, 1, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BasicNoDownsample(nn.Module):

    def __init__(self, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            basic.BasicBlock2d(3, 24, 3, 1, activation),
            basic.BasicBlock2d(24, 48, 3, 1, activation),
            basic.BasicBlock2d(48, 96, 3, 1, activation),
            basic.BasicBlock2d(96, 192, 3, 1, activation),
        )

        self.decoder = nn.Sequential(
            basic.BasicTransposeBlock2d(192, 96, 3, 1, activation),
            basic.BasicTransposeBlock2d(96, 48, 3, 1, activation),
            basic.BasicTransposeBlock2d(48, 24, 3, 1, activation),
            basic.BasicTransposeBlock2d(24, 3, 3, 1, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BasicNoDownscale(nn.Module):

    def __init__(self, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            basic.BasicBlock2d(3, 24, 3, 1, activation),
            basic.BasicBlock2d(24, 48, 3, 1, activation),
            basic.BasicBlock2d(48, 96, 3, 1, activation),
            basic.BasicBlock2d(96, 192, 3, 1, activation),
        )

        self.decoder = nn.Sequential(
            basic.BasicTransposeBlock2d(192, 96, 3, 1, activation),
            basic.BasicTransposeBlock2d(96, 48, 3, 1, activation),
            basic.BasicTransposeBlock2d(48, 24, 3, 1, activation),
            basic.BasicTransposeBlock2d(24, 3, 3, 1, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Resnet14(nn.Module):

    def __init__(self, n, residual_block_factory, downscale_factory, upscale_factory, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            transition.Transition(3, n, 3, activation),
            downscale_factory.get(n, n, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            downscale_factory.get(n, n*2, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            downscale_factory.get(n*2, n*4, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
        )

        self.decoder = nn.Sequential(
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            upscale_factory.get(n*4, n*2, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            upscale_factory.get(n*2, n, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            upscale_factory.get(n, n, activation),
            transition.Transition(n, 3, 3, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Resnet14NoDownscale(nn.Module):

    def __init__(self, n, residual_block_factory, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            transition.Transition(3, n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            transition.Transition(n, n*2, 1, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            transition.Transition(n*2, n*4, 1, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
        )

        self.decoder = nn.Sequential(
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            transition.Transition(n*4, n*2, 1, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            transition.Transition(n*2, n, 1, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            transition.Transition(n, 3, 3, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Resnet28(nn.Module):

    def __init__(self, n, residual_block_factory, downscale_factory, upscale_factory, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            transition.Transition(3, n, 3, activation),
            downscale_factory.get(n, n, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            downscale_factory.get(n, n*2, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            downscale_factory.get(n*2, n*4, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
        )

        self.decoder = nn.Sequential(
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            upscale_factory.get(n*4, n*2, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            upscale_factory.get(n*2, n, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            upscale_factory.get(n, n, activation),
            transition.Transition(n, 3, 3, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Resnet44(nn.Module):

    def __init__(self, n, residual_block_factory, downscale_factory, upscale_factory, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            transition.Transition(3, n, 3, activation),
            downscale_factory.get(n, n, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            downscale_factory.get(n, n*2, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            downscale_factory.get(n*2, n*4, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
        )

        self.decoder = nn.Sequential(
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            upscale_factory.get(n*4, n*2, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            upscale_factory.get(n*2, n, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            upscale_factory.get(n, n, activation),
            transition.Transition(n, 3, 3, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Resnet28NoDownscale(nn.Module):

    def __init__(self, n, residual_block_factory, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            transition.Transition(3, n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            transition.Transition(n, n*2, 1, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            transition.Transition(n*2, n*4, 1, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
        )

        self.decoder = nn.Sequential(
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            transition.Transition(n*4, n*2, 1, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            transition.Transition(n*2, n, 1, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            transition.Transition(n, 3, 3, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Resnet56NoDownscale(nn.Module):

    def __init__(self, n, residual_block_factory, activation='relu'):
        super().__init__()

        self.encoder = nn.Sequential(
            transition.Transition(3, n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            transition.Transition(n, n*2, 1, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            transition.Transition(n*2, n*4, 1, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
        )

        self.decoder = nn.Sequential(
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            residual_block_factory.get(n*4, 3, activation),
            transition.Transition(n*4, n*2, 1, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            residual_block_factory.get(n*2, 3, activation),
            transition.Transition(n*2, n, 1, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            residual_block_factory.get(n, 3, activation),
            transition.Transition(n, 3, 3, activation, final=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
