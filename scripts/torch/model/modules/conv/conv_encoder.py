from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, sample_length, sample_freqs, n_channels):
        super(ConvEncoder, self).__init__()
        self.sample_length = sample_length
        self.sample_freqs = sample_freqs
        self.n_channels = n_channels

        # self.conv1 = nn.Conv3d(1, 16, (18, 3, 3), stride=(1, 2, 2), padding="valid")
        self.conv1 = nn.Conv3d(1, 2, (n_channels, 3, 5), stride=(1, 1, 3), padding="valid")
        self.pool1 = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 0))
        self.bn1 = nn.BatchNorm3d(2)

        self.conv2 = nn.Conv3d(2, 4, (1, 3, 3), stride=(1, 1, 3), padding="valid")
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.bn2 = nn.BatchNorm3d(4)

        # self.conv3 = nn.Conv3d(32, 64, (1, 3, 3), stride=(1, 1, 1), padding="valid")
        # self.pool3 = nn.MaxPool3d((1, 2, 2))
        # self.bn3 = nn.BatchNorm3d(64)

    def forward(self, x):
        x = self.bn1(self.pool1(F.relu(self.conv1(x))))
        x = self.bn2(self.pool2(F.relu(self.conv2(x))))
        # x = self.bn3(self.pool3(F.relu(self.conv3(x))))
        return x
