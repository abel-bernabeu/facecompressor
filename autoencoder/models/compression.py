import torch
import torch.nn as nn
import autoencoder.models.quantization


class CompressionAutoencoder(torch.nn.Module):

    def __init__(self, quantize = True, num_bits = 8):
        super().__init__()

        self.encoder = None

        if quantize:
            self.quantize = autoencoder.models.Quantize()
            self.dequantize = autoencoder.models.Dequantize()
        else:
            self.quantize = None
            self.dequantize = None

        self.num_bits = num_bits

        self.decoder = None

    def forward(self, x):
        h = self.encoder(x)

        if self.quantize:
            batch_elems = x.size()[0]
            per_channel_num_bits = self.num_bits * torch.ones(batch_elems, self.encoder.hidden_state_num_channels).to(x.device)
            hq, per_channel_min, per_channel_max, per_channel_num_bits = self.quantize(h, quantization_select = None, per_channel_num_bits = per_channel_num_bits)
            h = self.dequantize(hq, per_channel_min, per_channel_max, per_channel_num_bits)

        y = self.decoder(h)

        yp = torch.nn.functional.hardtanh(y)

        return (yp + 1) * 0.5


class TwitterEncoder(torch.nn.Module):

    def __init__(self, hidden_state_num_channels):
        super().__init__()

        self.hidden_state_num_channels = hidden_state_num_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block7 = nn.Sequential(
            nn.Conv2d(128, self.hidden_state_num_channels, kernel_size=5, stride=2, padding=2,
                      padding_mode='replicate'))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) + x
        x = self.block4(x) + x
        x = self.block5(x) + x
        x = self.block6(x) + x
        x = self.block7(x)
        return x


class TwitterDecoder(torch.nn.Module):

    def __init__(self, hidden_state_num_channels):
        super().__init__()

        self.hidden_state_num_channels = hidden_state_num_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(self.hidden_state_num_channels, 512 * 4, kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU())

        self.block2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 256 * 4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.block7 = nn.Sequential(
            nn.Conv2d(256, 3 * 4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.PixelShuffle(2))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) + x
        x = self.block4(x) + x
        x = self.block5(x) + x
        x = self.block6(x)
        x = self.block7(x)
        return x


class TwitterCompressor(CompressionAutoencoder):

    def __init__(self, hidden_state_num_channels = 24, quantize = True, num_bits = 6):
        super().__init__(quantize = quantize, num_bits = num_bits)
        self.encoder = TwitterEncoder(hidden_state_num_channels = hidden_state_num_channels)
        self.decoder = TwitterDecoder(hidden_state_num_channels = hidden_state_num_channels)


class Compressor(TwitterCompressor):

    def __init__(self):
        super().__init__()
