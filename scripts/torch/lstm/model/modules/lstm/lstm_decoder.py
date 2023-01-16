from torch import nn


class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, encoding_dim, n_features):
        super(LSTMDecoder, self).__init__()

        self.seq_len = seq_len
        self.encoding_dim = encoding_dim
        self.hidden_dim = 2 * encoding_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=self.encoding_dim,
            hidden_size=self.encoding_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.encoding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(1, self.seq_len, 1)
        # x = x.reshape((1, self.seq_len, self.encoding_dim))

        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        # x = x.reshape((-1, self.seq_len, self.hidden_dim))
        return self.output_layer(x)
