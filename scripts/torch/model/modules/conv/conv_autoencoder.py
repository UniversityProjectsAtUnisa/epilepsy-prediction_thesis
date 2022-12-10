from torch import nn
from .conv_encoder import ConvEncoder
from .conv_decoder import ConvDecoder


class ConvAutoencoder(nn.Module):

    def __init__(self, sample_length, sample_freqs, n_channels):
        super(ConvAutoencoder, self).__init__()
        self.sample_length = sample_length
        self.sample_freqs = sample_freqs
        self.n_channels = n_channels
        self.encoder = ConvEncoder(sample_length, sample_freqs, n_channels)
        self.decoder = ConvDecoder(sample_length, sample_freqs, n_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
