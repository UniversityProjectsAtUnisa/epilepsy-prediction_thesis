import torch
from torch import nn


class Standardizer(nn.Module):
    def __init__(self):
        super(Standardizer, self).__init__()
        self.s = FunctionalStandardizer()

    def forward(self, x):
        return self.s.transform(x)

    def reset(self):
        self.s.reset()


class FunctionalStandardizer():
    def __init__(self):
        self.reset()

    def fit(self, x: torch.Tensor, axis=1):
        dim = list(range(len(x.shape)))
        dim.remove(axis)
        dim = tuple(dim)
        if not dim:
            dim = None
        self.mean = x.mean(dim=dim, keepdim=True)
        self.std = x.std(dim=dim, keepdim=True, unbiased=False)

    def transform(self, x: torch.Tensor):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted yet")
        return (x - self.mean) / self.std

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def reset(self):
        self.mean = None
        self.std = None
