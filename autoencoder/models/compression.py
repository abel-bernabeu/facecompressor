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