import pathlib

import torch


class Standardizer():
    mean_filename = "mean.pth"
    std_filename = "std.pth"

    def __init__(self):
        self.mean = None
        self.std = None

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

    @classmethod
    def load(cls, dirpath: pathlib.Path):
        s = cls()
        s.mean = torch.load(dirpath.joinpath(cls.mean_filename))
        s.std = torch.load(dirpath.joinpath(cls.std_filename))
        return s

    def save(self, dirpath: pathlib.Path):
        dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(self.mean, dirpath.joinpath(self.mean_filename))
        torch.save(self.std, dirpath.joinpath(self.std_filename))
