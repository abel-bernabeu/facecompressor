import torch
import torch.nn as nn
import autoencoder.models.quantization


class CompressionAutoencoder(torch.nn.Module):

    def __init__(self, quantization=False):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = None
        if quantization:
            self.quantize = autoencoder.models.Quantize()
            self.dequantize = autoencoder.models.Dequantize()
        else:
            self.quantize = None
            self.dequantize = None
        self.decoder = None

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        yp = torch.nn.functional.hardtanh(y)
        return (yp + 1) * 0.5


class QuantizingCompressionAutoencoder(torch.nn.Module):

    def __init__(self, num_bits):
        super(QuantizingCompressionAutoencoder, self).__init__()
        self.encoder = None
        self.num_bits = num_bits
        self.quantize = autoencoder.models.Quantize()
        self.dequantize = autoencoder.models.Dequantize()
        self.decoder = None

    def forward(self, x):
        h = self.encoder(x)

        batch_dim_index = 0
        channels_dim_index = 1
        rows_dim_index = 2
        cols_dim_index = 3

        batch = x.size()[batch_dim_index]
        channels  = x.size()[channels_dim_index]
        height = x.size()[rows_dim_index]
        width  = x.size()[cols_dim_index]

        per_channel_num_bits = self.num_bits * torch.ones(batch, self.encoder.hidden_state_num_channels).to(x.device)
        hq, per_channel_min, per_channel_max, per_channel_num_bits = self.quantize(h, quantization_select = None, per_channel_num_bits = per_channel_num_bits)
        hd = self.dequantize(hq, per_channel_min, per_channel_max, per_channel_num_bits)

        y = self.decoder(hd)

        yp = torch.nn.functional.hardtanh(y)

        return (yp + 1) * 0.5


class TwitterEncoder(torch.nn.Module):

    def __init__(self, hidden_state_num_channels):
        super(TwitterEncoder, self).__init__()

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
        super(TwitterDecoder, self).__init__()

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

    def __init__(self, hidden_state_num_channels = 96):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = TwitterEncoder(hidden_state_num_channels = hidden_state_num_channels)
        self.decoder = TwitterDecoder(hidden_state_num_channels = hidden_state_num_channels)