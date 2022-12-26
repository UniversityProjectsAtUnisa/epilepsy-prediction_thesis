import numpy as np
import numpy as np
import torch
import h5py
from sklearn.model_selection import train_test_split, KFold
import torch_config as config
from typing import List, Tuple, Optional
from itertools import islice
import pandas as pd


def is_consecutive(l):
    n = len(l)
    ans = n * min(l) + n * (n - 1) / 2 if n > 0 else 0
    return sum(l) == ans


def split_data(X_normal: Tuple[np.ndarray], train_size=0.8, random_state=1):
    print("Splitting data... ", end=" ")
    X_train, X_test = train_test_split(X_normal, train_size=train_size, random_state=random_state)  # type: ignore
    X_train, X_val = train_test_split(X_train, train_size=train_size, random_state=random_state)  # type: ignore
    print("DONE")
    return np.concatenate(X_train), np.concatenate(X_val), np.concatenate(X_test)


def convert_to_tensor(*Xs: np.ndarray) -> Tuple[torch.Tensor, ...]:
    return tuple([torch.tensor(x).float() for x in Xs])


def preprocess_data(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=1)


def load_patient_names(dataset_path) -> List[str]:
    with h5py.File(dataset_path) as f:
        return list(f.keys())


def load_data(dataset_path, patient_name, load_train=True, load_test=True, preprocess=True) -> Tuple[Optional[Tuple[np.ndarray]], Optional[Tuple[np.ndarray]]]:
    X_train = None
    X_test = None
    with h5py.File(dataset_path) as f:
        if load_train:
            print("Loading training data... ", end=" ")
            if preprocess:
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


def load_x_data(dataset_path, key, n_subwindows, preprocess) -> Tuple[np.ndarray]:
    x_data = []
    with h5py.File(dataset_path) as f:
        for x in f[key].values():  # type: ignore
            x = x[x.shape[0] % n_subwindows:]
            if preprocess:
                x = preprocess_data(x)
                x = x.reshape(-1, x.shape[-1]*n_subwindows)
            else:
                x = np.concatenate(x, -1)
                x = x.reshape(-1, x.shape[1], x.shape[-1]*n_subwindows)
            x_data.append(x)
    return tuple(x_data)


def load_y_data(dataset_path, key, n_subwindows) -> Tuple[pd.DataFrame]:
    with h5py.File(dataset_path) as f:
        sequence_ids = f[key].keys()  # type: ignore
    y_data = []
    for sid in sequence_ids:
        y = pd.read_hdf(dataset_path, f"{key}/{sid}")
        y = y[y.shape[0] % n_subwindows:]
        y_data.append(y)
    return tuple(y_data)


def load_numpy_dataset(
        dataset_path, patient_name, load_train=True, load_test=True, n_subwindows=2, preprocess=True) -> Tuple[
        Optional[Tuple[np.ndarray]],
        Optional[Tuple[np.ndarray]]]:
    X_train = None
    X_test = None
    if load_train:
        print("Loading training data... ", end=" ")
        X_train = load_x_data(dataset_path, f"{patient_name}/train/X", n_subwindows, preprocess)
        print("DONE")
        print(f"Training recordings: {len(X_train)}")
        print(f"Total training samples: {sum(x.shape[0] for x in X_train)}")

    if load_test:
        print("Loading test data... ", end=" ")
        X_test = load_x_data(dataset_path, f"{patient_name}/test/X", n_subwindows, preprocess)
        print(f'DONE')
        print(f"Testing recordings: {len(X_test)}")
        print(f"Total testing samples: {sum(x.shape[0] for x in X_test)}")
    return X_train, X_test  # type: ignore


def nested_kfolds(X: Tuple[np.ndarray], shuffle=False, random_state=None):
    n_elements = len(X)
    external_nfolds = min(n_elements, 5)
    e_kf = KFold(n_splits=external_nfolds, shuffle=shuffle, random_state=random_state)
    i_kf = KFold(n_splits=external_nfolds-1, shuffle=shuffle, random_state=random_state)

    for ei, (internal_idx, test_idx) in enumerate(e_kf.split(X)):
        X_internal = tuple(X[i] for i in internal_idx)
        X_test = tuple(X[i] for i in test_idx)
        for ii, (train_idx, val_idx) in enumerate(i_kf.split(X_internal)):
            X_train = tuple(X_internal[i] for i in train_idx)
            X_val = tuple(X_internal[i] for i in val_idx)

            yield ei, ii, (np.concatenate(X_train), np.concatenate(X_val), np.concatenate(X_test))


def patient_generic_kfolds(Xs: List[np.ndarray], shuffle=False, random_state=None):
    kf = KFold(n_splits=min(len(Xs), 5), shuffle=shuffle, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(kf.split(Xs)):
        X_train = [np.concatenate(Xs[k]) for k in train_idx]
        X_val = [np.concatenate(Xs[k]) for k in val_idx]
        yield i, (np.concatenate(X_train), np.concatenate(X_val))
