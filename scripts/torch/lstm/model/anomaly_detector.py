import pathlib
from typing import Optional

import torch
from ... import torch_config as config
from ...utils.gpu_utils import device_context

from .autoencoder import Autoencoder
from .threshold import Threshold


class AnomalyDetector:
    model_dirname = "model"
    threshold_filename = "threshold.pkl"

    @classmethod
    def load(cls, dirpath: pathlib.Path):
        model_path = dirpath.joinpath(cls.model_dirname)
        threshold_path = dirpath.joinpath(cls.threshold_filename)

        model = Autoencoder.load(model_path)
        threshold = Threshold.load(threshold_path)
        return cls(model, threshold)

    def __init__(self, model: Optional[Autoencoder] = None, threshold: Optional[Threshold] = None):
        self.model = model
        self.threshold = threshold

    def train(self, X_train, X_val, n_epochs, batch_size=64, dirpath: pathlib.Path = pathlib.Path("/tmp"), learning_rate=1e-3, plot_result=False):
        # Train model
        if self.model is None:
            n_channels = None
            sample_length = X_train.shape[1]
            self.model = Autoencoder(sample_length, config.N_SUBWINDOWS, n_channels)
        history = self.model.train_model(X_train, X_val, n_epochs=n_epochs, batch_size=batch_size, dirpath=dirpath.joinpath(
            self.model_dirname), learning_rate=learning_rate, plot_result=plot_result)

        # Calculate threshold
        losses_val = self.model.calculate_losses(X_val)
        self.threshold = Threshold()
        self.threshold.fit(losses_val)
        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        was_on_cpu = not X.is_cuda
        X.to(device_context.device)
        if self.model is None or self.threshold is None:
            raise Exception("Model not trained nor loaded")

        losses = self.model.calculate_losses(X)
        preds = self.threshold.transform(losses)
        if was_on_cpu:
            preds = preds.cpu()
        return preds

    def save(self, dirpath: pathlib.Path):
        if self.model is None or self.threshold is None:
            raise Exception("Model not trained nor loaded")
        model_path = dirpath.joinpath(self.model_dirname)
        threshold_path = dirpath.joinpath(self.threshold_filename)
        self.model.save(model_path)
        self.threshold.save(threshold_path)
