from torch import nn


class ConvDecoder(nn.Module):
    def __init__(self, sample_length, n_channels, n_filters, kernel_size):
        super(ConvDecoder, self).__init__()
        self.sample_length = sample_length
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv = nn.ConvTranspose2d(self.n_filters, 1, (self.kernel_size, self.n_channels), padding=((self.kernel_size-1//2), 0))

    def forward(self, x):
        x = self.conv(x)
        return x
