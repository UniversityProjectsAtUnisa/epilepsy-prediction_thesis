from torch import nn
from lstm.lstm_encoder import LSTMEncoder
from lstm.lstm_decoder import LSTMDecoder

N_SUBWINDOWS = 12
LEN_SUBWINDOWS = 128
ENCODING_DIM = LEN_SUBWINDOWS//2


class LSTMAutoencoder(nn.Module):

    def __init__(self, seq_len=N_SUBWINDOWS, n_features=LEN_SUBWINDOWS, encoding_dim=ENCODING_DIM):
        self.encoder = LSTMEncoder(seq_len, n_features, encoding_dim)
        self.decoder = LSTMDecoder(seq_len, encoding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
