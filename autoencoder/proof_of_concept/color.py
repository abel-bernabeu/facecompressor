import torch.nn as nn
import torch


class Colorize(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        x_r = x[:, 0, :, :]
        x_g = x[:, 1, :, :]
        x_b = x[:, 2, :, :]

        x_y = x_r * 0.299 + x_g * 0.587 + x_b * 0.114

        y = self.model(x)

        y_r = y[:, 0, :, :]
        y_g = y[:, 1, :, :]
        y_b = y[:, 2, :, :]

        y_cb = 0.5 - y_r * 0.168736 - y_g * 0.331264 + y_b * 0.5
        y_cr = 0.5 + y_r * 0.5 - y_g * 0.418688 + y_b * 0.081312

        y_r = x_y + (y_cr - 0.5) * 1.402
        y_g = x_y - (y_cb - 0.5) * 0.344136 - (y_cr - 0.5) * 0.714136
        y_b = x_y + (y_cb - 0.5) * 1.772

        y = torch.stack((y_r, y_g, y_b), dim=1)

        return y
