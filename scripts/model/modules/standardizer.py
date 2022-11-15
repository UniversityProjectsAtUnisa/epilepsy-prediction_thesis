import torch
from torch import nn


class Standardizer():
    def __init__(self):
        self.reset()

    def fit(self, x: torch.Tensor, axis=1):
        dim = list(x.shape)
        dim.remove(axis)
        dim = tuple(dim)
        if not dim:
            dim = None
        self.mean = x.mean(dim=dim)
        self.std = x.std(dim=dim, unbiased=False)

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted yet")
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def reset(self):
        self.mean = None
        self.std = None
