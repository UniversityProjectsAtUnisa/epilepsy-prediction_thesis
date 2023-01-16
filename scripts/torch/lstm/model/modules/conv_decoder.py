from torch import nn


class ConvDecoder(nn.Module):
    def __init__(self, sample_length, n_channels):
        super(ConvDecoder, self).__init__()
        self.sample_length = sample_length
        self.n_channels = n_channels

        self.conv = nn.ConvTranspose2d(1, 1, (self.n_channels, 1))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 1, -1)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1, x.shape[3])
        return x
