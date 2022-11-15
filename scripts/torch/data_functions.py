import numpy as np
# import pandas as pd
# from typing import Any
# import os
import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import classification_report, confusion_matrix
# from skimage.filters import threshold_otsu
# from typing import Any
import torch
import h5py
from sklearn.model_selection import train_test_split


def is_consecutive(l):
    n = len(l)
    ans = n * min(l) + n * (n - 1) / 2 if n > 0 else 0
    return sum(l) == ans


def split_data(X: np.ndarray, test_size=0.2) -> tuple[np.ndarray, np.ndarray]:
    print("Splitting data... ", end=" ")
    data: tuple[np.ndarray, np.ndarray] = train_test_split(X, test_size=test_size)  # type: ignore
    X_train, X_test = data
    return X_train, X_test


def convert_to_tensor(X):
    return torch.from_numpy(X).float()


def load_data(dataset_path, patient_name):

    print("Loading data... ", end=" ")

    with h5py.File(dataset_path) as f:    # TODO: Cambiare
        X: np.ndarray = f[f"{patient_name}/normal"][:500]  # type: ignore
        # X: np.ndarray = f[f"{patient_name}/normal"][:]  # type: ignore

    print(f'DONE  Shape: {X.shape}')

    return X
