from torch import nn
import torch.nn.functional as F
import torch


def build_padded_conv(in_ch, out_ch, kernel):
    padding = (kernel // 2 + (kernel - 2 * (kernel // 2)) - 1, kernel // 2)
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, padding=padding)


class FeatureExtractor(nn.Module):
    def __init__(self, lead_count: int):
        super(FeatureExtractor, self).__init__()

        self.conv1 = build_padded_conv(in_ch=lead_count, out_ch=1024, kernel=5)
        self.bn1 = nn.BatchNorm2d(1024)

        self.conv2 = build_padded_conv(in_ch=1024, out_ch=512, kernel=5)
        self.bn2 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.1)

        self.conv3 = build_padded_conv(in_ch=512, out_ch=256, kernel=3)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = build_padded_conv(in_ch=256, out_ch=128, kernel=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.drop4 = nn.Dropout(0.1)

        self.conv5 = build_padded_conv(in_ch=128, out_ch=64, kernel=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.drop5 = nn.Dropout(0.1)

        self.conv6 = build_padded_conv(in_ch=64, out_ch=32, kernel=3)
        self.bn6 = nn.BatchNorm2d(32)
        self.drop6 = nn.Dropout(0.1)

        self.conv7 = build_padded_conv(in_ch=32, out_ch=16, kernel=3)
        self.bn7 = nn.BatchNorm2d(16)
        self.drop7 = nn.Dropout(0.1)

        self.conv8 = build_padded_conv(in_ch=16, out_ch=8, kernel=3)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))

        x = self.bn2(F.relu(self.conv2(x)))
        x = self.drop2(self.pool2(x))

        x = self.bn3(F.relu(self.conv3(x)))

        x = self.bn4(F.relu(self.conv4(x)))
        x = self.drop4(self.pool4(x))

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.drop5(x)

        x = self.bn6(F.relu(self.conv6(x)))
        x = self.drop6(x)

        x = self.bn7(F.relu(self.conv7(x)))
        x = self.drop7(x)

        x = F.relu(self.conv8(x))
        x = self.flatten(x)

        return x
