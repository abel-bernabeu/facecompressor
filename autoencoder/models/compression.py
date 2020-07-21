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
        return self.operator(x)



class IdentityCompressor(CompressionAutoencoder):

    def __init__(self, input_width, input_height):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = nn.Identity(input_width, input_height)
        self.decoder = nn.Identity(input_width, input_height)


class MockCompressor(CompressionAutoencoder):

    def __init__(self, input_width, input_height):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = MockEncoder(input_width, input_height)
        self.decoder = MockDecoder(input_width, input_height)