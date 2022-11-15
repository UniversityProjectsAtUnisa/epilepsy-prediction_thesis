from torch import nn

N_SUBWINDOWS = 12
LEN_SUBWINDOWS = 128
ENCODING_DIM = LEN_SUBWINDOWS//2


class LSTMEncoder(nn.Module):
    def __init__(self, seq_len=N_SUBWINDOWS, n_features=LEN_SUBWINDOWS, encoding_dim=ENCODING_DIM):
        super(LSTMEncoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.hidden_dim = 2 * encoding_dim

        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.encoding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        _, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((1, self.encoding_dim))
