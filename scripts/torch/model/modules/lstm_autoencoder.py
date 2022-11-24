from torch import nn
from .lstm.lstm_encoder import LSTMEncoder
from .lstm.lstm_decoder import LSTMDecoder


class LSTMAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, encoding_dim):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.encoder = LSTMEncoder(seq_len, n_features, encoding_dim)
        self.decoder = LSTMDecoder(seq_len, encoding_dim, n_features)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.seq_len, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(x.shape[0], 1, -1)
        return x
