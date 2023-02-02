import pathlib
from typing import Optional

import torch
from ...utils.gpu_utils import device_context

from sklearn.svm import OneClassSVM
import numpy as np

from .modules.feature_extractor import FeatureExtractor
import joblib as jl


class CNNAnomalyDetector:
    model_dirname = "cnnanomalydetector"
    fe_filename = "feature_extractor.pth"
    svm_filename = "svm.joblib"

    def __init__(self, fe: FeatureExtractor, svm: Optional[OneClassSVM] = None):
        self.fe = fe
        self.svm = svm if svm is not None else OneClassSVM(cache_size=10000, nu=0.01, kernel="rbf", gamma="scale")

    @classmethod
    def load(cls, fe_dirpath: pathlib.Path, svm_dirpath: Optional[pathlib.Path] = None):
        fe = FeatureExtractor.load(fe_dirpath/cls.fe_filename)
        for p in fe.parameters():
            p.requires_grad = False
        if not svm_dirpath:
            return cls(fe)
        svm = jl.load(svm_dirpath/cls.svm_filename)
        if not isinstance(svm, OneClassSVM):
            raise Exception("Loaded model is not a OneClassSVM")
        return cls(fe, svm)

    def load_svm(self, svm_dirpath: pathlib.Path):
        svm = jl.load(svm_dirpath/self.svm_filename)
        if not isinstance(svm, OneClassSVM):
            raise Exception("Loaded model is not a OneClassSVM")
        self.svm = svm

    def save_svm(self, dirpath: pathlib.Path):
        dirpath.mkdir(parents=True, exist_ok=True)
        jl.dump(self.svm, dirpath/self.svm_filename)

    def train(self, embeddings: np.ndarray):
        return self.svm.fit(embeddings)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        embeddings = self.calculate_embeddings(X)
        if self.svm is None:
            raise Exception("Model not trained nor loaded")
        return torch.Tensor(self.svm.predict(embeddings) < 0)

    def calculate_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        was_on_cpu = not X.is_cuda
        X.to(device_context.device)
        if self.fe is None:
            raise Exception("Model not trained nor loaded")
        self.fe.eval()
        embeddings = self.fe(X)
        if was_on_cpu:
            embeddings = embeddings.cpu()
        return embeddings
