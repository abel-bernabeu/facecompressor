import torch.nn as nn


class Downscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool2d(2, 2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        X = self.conv(X)
        X = self.norm(X)
        X = self.activation(X)
        X = self.pool(X)
        return X


class Encoder(nn.Module):
    def __init__(self, dimension=128):
        super(Encoder, self).__init__()
        self.down1 = Downscale(3, 48, 7)
        self.down2 = Downscale(48, 64, 3)
        self.down3 = Downscale(64, 128, 3)
        self.fc = nn.Linear(128 * 4 * 4, dimension)
        # self.activation = nn.ELU()

    def forward(self, X):
        X = self.down1(X)
        X = self.down2(X)
        X = self.down3(X)
        X = X.reshape(X.shape[0], -1)
        X = self.fc(X)
        return X  # self.activation(X)


class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upscale, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1)
        self.activation = nn.ELU()
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.BatchNorm2d(out_channels * 4)

    def forward(self, X):
        X = self.conv(X)
        X = self.norm(X)
        X = self.activation(X)
        X = self.shuffle(X)
        return X


class Decoder(nn.Module):
    def __init__(self, dimension=128):
        super(Decoder, self).__init__()
        self.up0 = nn.Linear(dimension, 4 * 4 * 128)
        self.up1 = Upscale(128, 64)
        self.up2 = Upscale(64, 32)
        self.up3 = Upscale(32, 16)
        self.conv = nn.Conv2d(16, 3, 3, 1, 1, 1)

    def forward(self, X):
        X = self.up0(X)
        X = X.reshape(X.shape[0], 128, 4, 4)
        X = self.up1(X)
        X = self.up2(X)
        X = self.up3(X)
        X = self.conv(X)
        return X


class ConvolutionalAE(nn.Module):
    def __init__(self, dimension=128):
        super(ConvolutionalAE, self).__init__()
        self.encoder = Encoder(dimension)
        self.decoder = Decoder(dimension)

    def forward(self, X):
        X_enc = self.encoder(X)
        X_reconstructed = self.decoder(X_enc)
        return X_reconstructed
