import pickle
import numpy as np
import torch
import pathlib


class Threshold:
    def __init__(self, threshold: float):
        self.threshold = threshold

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            threshold = pickle.load(f)

        if not isinstance(threshold, cls):
            raise Exception("Invalid threshold file")
        return threshold

    def save(self, filepath: pathlib.Path):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def __call__(self, x: torch.Tensor):
        return x > self.threshold
