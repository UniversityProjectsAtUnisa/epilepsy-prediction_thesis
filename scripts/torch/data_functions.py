import numpy as np
# import pandas as pd
# from typing import Any
# import os
import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import classification_report, confusion_matrix
# from typing import Any
import torch
import h5py
from sklearn.model_selection import train_test_split
import torch_config as config
from typing import List


def is_consecutive(l):
    n = len(l)
    ans = n * min(l) + n * (n - 1) / 2 if n > 0 else 0
    return sum(l) == ans


# def split_data(X: np.ndarray, test_size=0.2) -> tuple[np.ndarray, np.ndarray]:
#     print("Splitting data... ", end=" ")
#     data: tuple[np.ndarray, np.ndarray] = train_test_split(X, test_size=test_size)  # type: ignore
#     X_train, X_test = data
#     return X_train, X_test

def split_data(X_normal: np.ndarray, X_anomalies: List[np.ndarray],
               train_size=0.8, random_state=1) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    print("Splitting data... ", end=" ")
    X_train, rest = train_test_split(X_normal, train_size=train_size, random_state=1)  # type: ignore
    n_anomalies = sum([x.shape[0] for x in X_anomalies])
    if n_anomalies > len(rest):
        raise ValueError("Not enough data for anomalies")
    X_val, X_test = train_test_split(rest, test_size=n_anomalies, random_state=1)
    return X_train, X_val, X_test, X_anomalies  # type: ignore


def convert_to_tensor(*Xs: np.ndarray) -> tuple[torch.Tensor, ...]:
    return tuple([torch.tensor(x).float() for x in Xs])


def load_data(dataset_path, patient_name):

    print("Loading data... ", end=" ")

    with h5py.File(dataset_path) as f:    # TODO: Cambiare
        X_normal: np.ndarray = f[f"{patient_name}/normal"][:config.PARTIAL_TRAINING]  # type: ignore
        # X: np.ndarray = f[f"{patient_name}/normal"][:]  # type: ignore
        X_anomalies: List[np.ndarray] = []
        i = 0
        for dataset in f[f"{patient_name}/anomaly"].values():  # type: ignore
            X_anomalies.append(dataset[:])
            i += 1
            if i == 1:
                break

    print(f'DONE')
    print(f"Normal shape: {X_normal.shape}")
    print(f"Anomalies shapes:")
    print('\n'.join([str(x.shape) for x in X_anomalies]))
    print(f"Total anomalies: {sum([x.shape[0] for x in X_anomalies])}")

    return X_normal, X_anomalies
