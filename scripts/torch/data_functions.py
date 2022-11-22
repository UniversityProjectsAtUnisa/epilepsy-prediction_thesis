import numpy as np
import numpy as np
import torch
import h5py
from sklearn.model_selection import train_test_split
import torch_config as config
from typing import List, Tuple, Optional
from itertools import islice


def is_consecutive(l):
    n = len(l)
    ans = n * min(l) + n * (n - 1) / 2 if n > 0 else 0
    return sum(l) == ans


def split_data(X_normal: Tuple[np.ndarray], train_size=0.8, random_state=1):
    print("Splitting data... ", end=" ")
    X_train, X_val = train_test_split(X_normal, train_size=train_size, random_state=random_state)  # type: ignore
    print("DONE")
    return np.concatenate(X_train), np.concatenate(X_val)


def convert_to_tensor(*Xs: np.ndarray) -> Tuple[torch.Tensor, ...]:
    return tuple([torch.tensor(x).float() for x in Xs])


def preprocess_data(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=1)


def load_data(dataset_path, patient_name, load_train=True, load_test=True, preprocess=True) -> Tuple[Optional[Tuple[np.ndarray]], Optional[Tuple[np.ndarray]]]:
    X_train = None
    X_test = None
    with h5py.File(dataset_path) as f:
        if load_train:
            print("Loading training data... ", end=" ")
            if preprocess_data:
                X_train_generator = (preprocess_data(x[:]) for x in f[f"{patient_name}/train"].values())  # type: ignore
            else:
                X_train_generator = (x[:] for x in f[f"{patient_name}/train"].values())  # type: ignore
            if config.PARTIAL_TRAINING == 0:
                X_train = tuple(X_train_generator)
            else:
                X_train = tuple(islice(X_train_generator, config.PARTIAL_TRAINING))
            print("DONE")
            print(f"Training recordings: {len(X_train)}")
            print(f"Total training samples: {sum(x.shape[0] for x in X_train)}")

        if load_test:
            print("Loading test data... ", end=" ")
            if preprocess:
                X_test_generator = (preprocess_data(x[:]) for x in f[f"{patient_name}/test"].values())  # type: ignore
            else:
                X_test_generator = (x[:] for x in f[f"{patient_name}/test"].values())  # type: ignore
            if config.PARTIAL_TESTING == 0:
                X_test = tuple(X_test_generator)
            else:
                X_test = tuple(islice(X_test_generator, config.PARTIAL_TESTING))
            print(f'DONE')
            print(f"Testing recordings: {len(X_test)}")
            print(f"Total testing samples: {sum(x.shape[0] for x in X_test)}")

    return X_train, X_test
