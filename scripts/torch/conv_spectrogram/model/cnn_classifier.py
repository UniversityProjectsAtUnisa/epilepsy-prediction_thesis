from .modules.feature_extractor import FeatureExtractor
from .modules.dense_classifier import DenseClassifier

from torch import nn


class CnnClassifier(nn.Module):
    def __init__(self, lead_count: int):
        super(CnnClassifier, self).__init__()

        self.fe = FeatureExtractor(lead_count)
        self.mlp = DenseClassifier()

    def forward(self, x):
        return self.mlp(self.fe(x))
