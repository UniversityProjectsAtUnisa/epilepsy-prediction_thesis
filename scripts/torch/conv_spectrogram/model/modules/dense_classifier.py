from torch import nn
import torch.nn.functional as F
import torch
import pathlib
from ....utils.gpu_utils import device_context


class DenseClassifier(nn.Module):
    def __init__(self):
        super(DenseClassifier, self).__init__()
        self.fc1 = nn.Linear(80, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x

    @classmethod
    def load(cls, filepath: pathlib.Path):
        if not filepath.exists():
            raise Exception("Model file not found")
        model = torch.load(filepath, map_location=device_context.device)
        if not isinstance(model, cls):
            raise Exception("Invalid model file")
        return model

    def save(self, filepath: pathlib.Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, filepath)
