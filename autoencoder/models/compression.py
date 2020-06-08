import torch
import autoencoder.models.quantization


class CompressionAutoencoder(torch.nn.Module):

    def __init__(self, encoder, decoder, input_width, input_height):
        super(CompressionAutoencoder, self).__init__()
        self.encoder = encoder
        self.quantize = autoencoder.models.Quantize()
        self.dequantize = autoencoder.models.Dequantize()
        self.decoder = decoder

    def forward(self, x, quantization_select, per_channel_num_bits):
        h = self.encoder(x)
        hq, per_channel_min, per_channel_max, _ = self.quantize(h, quantization_select, per_channel_num_bits)
        hdq = self.dequantize(hq, per_channel_min, per_channel_max, per_channel_num_bits)
        return self.decoder(hdq)


class Downscale2x2(torch.nn.Module):

    def __init__(self, input_width, input_height):
        super(Downscale2x2, self).__init__()
        # TODO put a line here as part of issue #1

    def forward(self, x):
        """
        Applies a convolution in 2x2 blocks to an RGB image, so it gets
        downsized to half in height and width.
        """
        # TODO put a line here as part of issue #1
        return x


class Upscale2x2(torch.nn.Module):

    def __init__(self, input_width, input_height):
        super(Upscale2x2, self).__init__()
        # TODO put a line here as part of issue #1

    def forward(self, x):
        """
        Upscales in 2x2 blocks an RGB image, so it gets
        doubled in height and width.
        """
        # TODO put a line here as part of issue #1
        return x


class MockAutoencoder(CompressionAutoencoder):

    def __init__(self, input_width, input_height):
        encoder = Downscale2x2(input_width, input_height)
        decoder = Upscale2x2(input_width, input_height)
        super(CompressionAutoencoder, self), encoder, decoder, input_width, input_height.__init__()


def test_mockautoencoder():
    """
    A test instantiating the MockAutoencoder just to have some minimal confidence the
    super methods get sensible parameters passed.
    """
    autoencoder = MockAutoencoder(input_width=256, input_height=256)