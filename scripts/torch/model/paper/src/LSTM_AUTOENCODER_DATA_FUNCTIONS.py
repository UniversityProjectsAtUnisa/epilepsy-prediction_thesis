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


# def load_data(dataset_path, patient_name, load_train=True, load_test=True, preprocess=True) -> Tuple[Optional[Tuple[np.ndarray]],
#                                                                                                      Optional[np.ndarray],
#                                                                                                      Optional[np.ndarray]]:
#     X_train = None
#     X_test_normal = None
#     X_test_ictal = None
#     with h5py.File(dataset_path) as f:
#         if load_train:
#             print("Loading training data... ", end=" ")
#             if preprocess:
#                 X_train_generator = (preprocess_data(x[:]) for x in f[f"{patient_name}/train"].values())  # type: ignore
#             else:
#                 X_train_generator = (x[:] for x in f[f"{patient_name}/train"].values())  # type: ignore
#             X_train = tuple(X_train_generator)
#             print("DONE")
#             print(f"Training recordings: {len(X_train)}")
#             print(f"Total training samples: {sum(x.shape[0] for x in X_train)}")

#         if load_test:
#             print("Loading test data... ", end=" ")
#             if preprocess:
#                 X_test_normal_generator = (preprocess_data(x[:], axis=0) for x in f[f"{patient_name}/test/normal"])  # type: ignore
#                 X_test_ictal_generator = (preprocess_data(x[:], axis=0) for x in f[f"{patient_name}/test/ictal"])  # type: ignore
#             else:
#                 X_test_normal_generator = (x[:] for x in f[f"{patient_name}/test/normal"])  # type: ignore
#                 X_test_ictal_generator = (x[:] for x in f[f"{patient_name}/test/ictal"])  # type: ignore
#             X_test_normal = np.array(tuple(X_test_normal_generator))
#             X_test_ictal = np.array(tuple(X_test_ictal_generator))
#             print(f'DONE')
#             print(f"Test normal recordings: {len(X_test_normal)}")
#             print(f"Test ictal recordings: {len(X_test_ictal)}")

#     return X_train, X_test_normal, X_test_ictal


def load_data(directory, x_name, y_name, conf_channel=3, pandas=False, seql=0):

    print("Loading x_data... ", end=" ")

    x_data = np.load(directory/x_name, allow_pickle=True)  # datos

    if conf_channel == 1:
        data_x = x_data[:, 5]-x_data[:, 9]  # resta del canal 6 y 10, ofrecen más información sobre ataques epilepticos
    elif conf_channel == 2:
        data_x = x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2])  # concatenacion 21 canales
    elif conf_channel == 3:
        data_x = x_data.mean(axis=1)  # media aritmetica canales

    else:
        print('ERROR LOAD TYPE!!')

    print(f'DONE  Shape: {x_data.shape}')

    print("Loading y_data... ", end=" ")
    y_data = np.load(directory/y_name, allow_pickle=True)  # metadatos
    data_y = y_data
    #data_y = y_data [:,7].astype('int')
    print(f'DONE  Shape: {y_data.shape}')

    print("Tranform data... ", end=" ")

    if pandas:
        print("Loading pandas data_x... ")
        data_x = pd.DataFrame(data_x)
        print("Loading pandas data_y... ")
        data_y = pd.DataFrame(data_y, columns=['type', 'name', 'filename', 'pre1', 'pre2', 'id_eeg_actual', 'id_eeg_all', 'label'])
        data_y.label = data_y.label.astype('int')

        #rus = RandomUnderSampler(sampling_strategy=0.5)
        #data_x_r, data_y_r= rus.fit_resample(data_x, data_y)

    # tranformamos [NSamples x 128] a [NSamples x LSeq x 128]
    if seql != 0:
        nSamp = int(data_x.shape[0]/seql)
        data_x = data_x[:nSamp*seql].reshape(-1, seql, 128)
        data_y = data_y[:nSamp*seql, -1].reshape(-1, seql)

    print(f'DONE  Shape: {data_x.shape} \n')

    return data_x, data_y
