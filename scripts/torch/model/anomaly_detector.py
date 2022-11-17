from .modules.standardizer import Standardizer
from .modules.threshold import Threshold
from .autoencoder import Autoencoder
import torch_config as config
import numpy as np
from skimage.filters import threshold_otsu
import pathlib
import torch
from utils import check_device
from typing import Optional


class AnomalyDetector:
    model_dirname = "model"
    standardizer_dirname = "standardizer"
    threshold_filename = "threshold.pkl"

    @classmethod
    def load(cls, dirpath: pathlib.Path):
        model_path = dirpath.joinpath(cls.model_dirname)
        standardizer_path = dirpath.joinpath(cls.standardizer_dirname)
        threshold_path = dirpath.joinpath(cls.threshold_filename)

        model = Autoencoder.load(model_path)
        standardizer = Standardizer.load(standardizer_path)
        threshold = Threshold.load(threshold_path)
        return cls(model, standardizer, threshold)

    def __init__(self, model: Optional[Autoencoder] = None, standardizer: Optional[Standardizer] = None, threshold: Optional[Threshold] = None):
        self.model = model
        self.standardizer = standardizer
        self.threshold = threshold
        self.device = check_device()  # TODO: Use device

    def train(self, X_train, X_val, n_epochs, batch_size=64, dirpath: pathlib.Path = pathlib.Path("/tmp"), learning_rate=1e-3, plot_result=False):
        self.standardizer = Standardizer()
        X_train = self.standardizer.fit_transform(X_train)
        X_val = self.standardizer.transform(X_val)

        sample_length = X_train.shape[2]
        n_channels = X_train.shape[1]
        self.model = Autoencoder(sample_length, config.N_FILTERS, n_channels, config.KERNEL_SIZE, config.N_SUBWINDOWS)
        self.model.train_model(X_train, X_val, n_epochs=n_epochs, batch_size=batch_size, dirpath=dirpath.joinpath(
            self.model_dirname), learning_rate=learning_rate, plot_result=plot_result)

        losses_train = self.model.calculate_losses(X_train)

        # calculate threshold
        th: float = threshold_otsu(np.array(losses_train.cpu()))
        self.threshold = Threshold(th)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.model is None or self.standardizer is None or self.threshold is None:
            raise Exception("Model not trained nor loaded")

        X = self.standardizer.transform(X)
        losses = self.model.calculate_losses(X)
        return self.threshold(torch.Tensor(losses))

    def save(self, dirpath: pathlib.Path):
        if self.model is None or self.standardizer is None or self.threshold is None:
            raise Exception("Model not trained nor loaded")
        model_path = dirpath.joinpath(self.model_dirname)
        standardizer_path = dirpath.joinpath(self.standardizer_dirname)
        threshold_path = dirpath.joinpath(self.threshold_filename)
        self.model.save(model_path)
        self.standardizer.save(standardizer_path)
        self.threshold.save(threshold_path)
