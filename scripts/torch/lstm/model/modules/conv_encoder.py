from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, sample_length, n_channels):
        super(ConvEncoder, self).__init__()
        self.sample_length = sample_length
        self.n_channels = n_channels

        self.conv = nn.Conv2d(1, 1, (self.n_channels, 1))

    def forward(self, x):
        # x = x.reshape((-1, 1, self.n_channels, self.sample_length))
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
        x = self.conv(x)
        x = x.reshape((x.shape[0], 1, -1))
        return x
