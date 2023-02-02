import numpy as np
import torch
import h5py
from sklearn.model_selection import train_test_split, KFold
from typing import List, Tuple, Optional, Union
import pandas as pd
import pathlib


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


def segment_data(X: np.ndarray, n_subwindows: int, overlap: int, preprocess: bool = False) -> np.ndarray:
    stride = n_subwindows - overlap
    skip = (len(X) - overlap) % stride

    segments = np.array(tuple(zip(*(X[skip+i::stride] for i in range(n_subwindows)))))
    segments = np.moveaxis(segments, 1, -1)
    if not preprocess:
        return segments
    return segments.reshape(segments.shape[0], -1)


def load_x_data(dataset_path, key, preprocess):
    x_data = []
    with h5py.File(dataset_path) as f:
        for x in f[key].values():  # type: ignore
            x = x[:]
            if preprocess:
                x = preprocess_data(x)
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


def load_data_for_traditional_training(
        dataset_path, patient_name, preictal_seconds, n_subwindows=6, overlap=5) -> Tuple[np.ndarray, np.ndarray]:

    print("Loading normal data... ", end=" ")
    X_train = load_x_data(dataset_path, f"{patient_name}/train/X", False)
    print("DONE")
    print(f"Normal recordings: {len(X_train)}")

    print("Loading seizure data... ", end=" ")
    X_test = load_x_data(dataset_path, f"{patient_name}/test/X", False)
    print(f'DONE')
    print(f"Seizure recordings: {len(X_test)}")

    X_train = [segment_data(x, n_subwindows, overlap) for x in X_train]

    positives = []
    for x in X_test:
        n, p = np.split(x, [-preictal_seconds])
        X_train.append(segment_data(n, n_subwindows, overlap))
        positives.append(segment_data(p, n_subwindows, overlap))

    y = np.concatenate([np.zeros(sum(len(x) for x in X_train)), np.ones(sum(len(x) for x in positives))])
    X_train.extend(positives)
    X = np.concatenate(X_train)
    print(f"Total normal samples: {len(y) - sum(y)}")
    print(f"Total seizure samples: {sum(y)}")

    return X, y  # type: ignore


def load_data(
        dataset_path, patient_name, load_train=True, load_test=True, n_subwindows=2, overlap=0, preprocess=True) -> Tuple[
        Optional[Tuple[np.ndarray]],
        Optional[Tuple[np.ndarray]]]:
    X_train = None
    X_test = None
    if load_train:
        print("Loading training data... ", end=" ")
        X_train = load_x_data(dataset_path, f"{patient_name}/train/X", preprocess)
        X_train = tuple(segment_data(x, n_subwindows, overlap, preprocess) for x in X_train)
        print("DONE")
        print(f"Training recordings: {len(X_train)}")
        print(f"Total training samples: {sum(x.shape[0] for x in X_train)}")

    if load_test:
        print("Loading test data... ", end=" ")
        X_test = load_x_data(dataset_path, f"{patient_name}/test/X", preprocess)
        X_test = tuple(segment_data(x, n_subwindows, overlap, preprocess) for x in X_test)
        print(f'DONE')
        print(f"Testing recordings: {len(X_test)}")
        print(f"Total testing samples: {sum(x.shape[0] for x in X_test)}")
    return X_train, X_test  # type: ignore


def nested_kfolds(X: Tuple[Union[np.ndarray, torch.Tensor]], shuffle=False, random_state=None):
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


def interpatient_nested_kfolds(X: Tuple[np.ndarray], input_patient_dirpath: pathlib.Path, shuffle=False, random_state=None):
    input_kf = (d for d in input_patient_dirpath.iterdir() if d.is_dir())

    for input_path in input_kf:
        for ei, ii, (X_train, X_val, X_test) in nested_kfolds(X, shuffle, random_state):
            yield input_path, ei, ii, (X_train, X_val, X_test)


def patient_generic_kfolds(Xs: List[np.ndarray], shuffle=False, random_state=None):
    kf = KFold(n_splits=min(len(Xs), 5), shuffle=shuffle, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(kf.split(Xs)):
        X_train = [np.concatenate(Xs[k]) for k in train_idx]
        X_val = [np.concatenate(Xs[k]) for k in val_idx]
        yield i, (np.concatenate(X_train), np.concatenate(X_val))


def patient_specific_kfolds_conv(Xs: Tuple[np.ndarray], shuffle=False, random_state=None):
    kf = KFold(n_splits=min(len(Xs), 5), shuffle=shuffle, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(kf.split(Xs)):
        X_train = [Xs[k] for k in train_idx]
        X_test = [Xs[k] for k in val_idx]
        yield i, (np.concatenate(X_train), np.concatenate(X_test))


def patient_generic_kfolds_h5py(Xs, ys, shuffle=False, random_state=None):
    kf = KFold(n_splits=min(len(Xs), 4), shuffle=shuffle, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(kf.split(Xs)):
        X_train = np.concatenate([Xs[f"{k}"] for k in train_idx])
        X_val = np.concatenate([(Xs[f"{k}"]) for k in val_idx])
        y_train = np.concatenate([(ys[f"{k}"]) for k in train_idx])
        y_val = np.concatenate([(ys[f"{k}"]) for k in val_idx])
        yield i, (X_train, y_train, X_val, y_val)
