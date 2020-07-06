import torch
import torch.nn as nn
import autoencoder.models.quantization


class CompressionAutoencoder(torch.nn.Module):

    def __init__(self):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = None
        self.quantize = autoencoder.models.Quantize()
        self.dequantize = autoencoder.models.Dequantize()
        self.decoder = None

    def forward(self, x): ##, quantization_select, per_channel_num_bits):
        h = self.encoder(x)
        #hq, per_channel_min, per_channel_max, _ = self.quantize(h, quantization_select, per_channel_num_bits)
        #hdq = self.dequantize(hq, per_channel_min, per_channel_max, per_channel_num_bits)
        y = self.decoder(h)
        yp = torch.tanh(y)
        return (yp + 1) * 0.5


class MockEncoder(torch.nn.Module):

    def __init__(self, input_width, input_height):
        super(MockEncoder, self).__init__()
        self.operator = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Applies a convolution in 2x2 blocks to an RGB image, so it gets
        downsized to half in height and width.
        """
        return self.operator(x)


class MockDecoder(torch.nn.Module):

    def __init__(self, input_width, input_height):
        super(MockDecoder, self).__init__()
        self.operator = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """
        Upscales in 2x2 blocks an RGB image, so it gets
        doubled in height and width.
        """
        return self.operator(x)


class MockCompressor(CompressionAutoencoder):

    def __init__(self, input_width, input_height):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = MockEncoder(input_width, input_height)
        self.decoder = MockDecoder(input_width, input_height)