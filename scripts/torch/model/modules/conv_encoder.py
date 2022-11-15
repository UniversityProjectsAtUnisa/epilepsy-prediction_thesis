from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, sample_length, n_channels, n_filters, kernel_size):
        super(ConvEncoder, self).__init__()
        self.sample_length = sample_length
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(1, self.n_filters, (self.n_channels, self.kernel_size), padding=(0, (self.kernel_size-1)//2))
        self.bn = nn.BatchNorm2d(self.n_filters)

    def forward(self, x):
        x = x.reshape((-1, 1, self.n_channels, self.sample_length))
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
