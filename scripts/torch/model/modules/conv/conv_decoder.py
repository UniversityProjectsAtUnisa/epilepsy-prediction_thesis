from torch import nn
import torch.nn.functional as F


class ConvDecoder(nn.Module):
    def __init__(self, sample_length, sample_freqs, n_channels):
        super(ConvDecoder, self).__init__()
        self.sample_length = sample_length
        self.sample_freqs = sample_freqs
        self.n_channels = n_channels

        self.conv1 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 1, 2), padding=(0, 0, 1), output_padding=(0, 0, 1))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.ConvTranspose3d(32, 16, (1, 3, 3), stride=(1, 1, 3))
        self.bn2 = nn.BatchNorm3d(16)

        self.conv3 = nn.ConvTranspose3d(16, 1, (n_channels, 3, 7), stride=(1, 2, 7), padding=(0, 0, 6))

        self.dense = nn.Linear(sample_length*sample_freqs, sample_length*sample_freqs)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.reshape((-1, 1, self.n_channels, self.sample_length*self.sample_freqs))
        x = self.dense(x)
        x = x.reshape((-1, 1, self.n_channels, self.sample_length, self.sample_freqs))
        return x
