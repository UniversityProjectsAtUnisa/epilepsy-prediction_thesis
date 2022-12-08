import os
import numpy as np
import pandas as pd
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from skimage.filters import threshold_otsu
import h5py
from typing import List, Tuple, Optional


def preprocess_data(X: np.ndarray, axis=1) -> np.ndarray:
    return X.mean(axis=axis)


def load_patient_names(dataset_path) -> List[str]:
    with h5py.File(dataset_path) as f:
        return list(f.keys())


def load_data(dataset_path, patient_name, load_train=True, load_test=True, preprocess=True) -> Tuple[Optional[Tuple[np.ndarray]],
                                                                                                     Optional[np.ndarray],
                                                                                                     Optional[np.ndarray]]:
    X_train = None
    X_test_normal = None
    X_test_ictal = None
    with h5py.File(dataset_path) as f:
        if load_train:
            print("Loading training data... ", end=" ")
            if preprocess:
                X_train_generator = (preprocess_data(x[:]) for x in f[f"{patient_name}/train"].values())  # type: ignore
            else:
                X_train_generator = (x[:] for x in f[f"{patient_name}/train"].values())  # type: ignore
            X_train = tuple(X_train_generator)
            print("DONE")
            print(f"Training recordings: {len(X_train)}")
            print(f"Total training samples: {sum(x.shape[0] for x in X_train)}")

        if load_test:
            print("Loading test data... ", end=" ")
            if preprocess:
                X_test_normal_generator = (preprocess_data(x[:], axis=0) for x in f[f"{patient_name}/test/normal"])  # type: ignore
                X_test_ictal_generator = (preprocess_data(x[:], axis=0) for x in f[f"{patient_name}/test/ictal"])  # type: ignore
            else:
                X_test_normal_generator = (x[:] for x in f[f"{patient_name}/test/normal"])  # type: ignore
                X_test_ictal_generator = (x[:] for x in f[f"{patient_name}/test/ictal"])  # type: ignore
            X_test_normal = np.array(tuple(X_test_normal_generator))
            X_test_ictal = np.array(tuple(X_test_ictal_generator))
            print(f'DONE')
            print(f"Test normal recordings: {len(X_test_normal)}")
            print(f"Test ictal recordings: {len(X_test_ictal)}")

    return X_train, X_test_normal, X_test_ictal
