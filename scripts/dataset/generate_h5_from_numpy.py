import copy
import json
import pathlib
from typing import Dict, List

import config
import h5py
import mne
import numpy as np
from tqdm import tqdm
from scipy import signal
import cv2
import pandas as pd
from collections import defaultdict
import warnings
from tables import NaturalNameWarning


def load_paper_labels(filepath: pathlib.Path):
    data_y = np.load(filepath, allow_pickle=True)  # metadatos
    data_y = pd.DataFrame(data_y, columns=['type', 'name', 'filename', 'pre1', 'pre2', 'id_eeg_actual', 'id_eeg_all', 'label'])
    data_y.label = data_y.label.astype('int')
    return data_y


def load_paper_data(filepath: pathlib.Path, lazy=False):
    x_data = np.load(filepath, allow_pickle=True, mmap_mode='r' if lazy else None)
    return x_data


def create_normal_datasets(h5_filepath, normal_x_filepath, normal_y_filepath):
    normal_y = load_paper_labels(normal_y_filepath)

    patient_idxs = defaultdict(list)
    patients = normal_y['name'].unique()
    for patient in patients:
        patient_normal_filenames = normal_y[normal_y["name"] == patient]['filename'].unique()
        for filename in patient_normal_filenames:
            idx = normal_y.index[normal_y['filename'] == filename]
            patient_idxs[patient].append(idx)

    normal_x = load_paper_data(normal_x_filepath, lazy=True)
    for patient, idxs in tqdm(patient_idxs.items(), desc="Normal data: Patients"):
        n_normals = 0
        for idx in tqdm(idxs, desc=f"{patient}", leave=False):
            with h5py.File(h5_filepath, 'a') as f:
                f.create_dataset(f"{patient}/train/X/{n_normals}", data=normal_x[idx])
            normal_y.iloc[idx].to_hdf(str(h5_filepath), key=f"{patient}/train/y/{n_normals}", mode="a")
            # f.create_dataset(f"{patient}/train/y/{n_normals}", data=normal_y.iloc[idx])
            n_normals += 1


def create_seizure_datasets(h5_filepath, seizure_x_filepath, seizure_y_filepath):
    seizure_y = load_paper_labels(seizure_y_filepath)

    patient_idxs = defaultdict(list)
    patients = seizure_y['name'].unique()
    for patient in patients:
        patient_seizure_filenames = seizure_y[seizure_y["name"] == patient]['filename'].unique()
        for filename in patient_seizure_filenames:
            filename_seizure_y = seizure_y[seizure_y['filename'] == filename]
            n_seizures = max(filename_seizure_y['pre1'])

            current_sequence_start = min(filename_seizure_y.index[filename_seizure_y['pre1'] == 0])
            current_sequence_end = min(filename_seizure_y.index[filename_seizure_y['pre1'] == 1])
            if current_sequence_end - current_sequence_start >= config.PREICTAL_SECONDS:
                patient_idxs[patient].append(np.arange(current_sequence_start, current_sequence_end))
            for i in range(2, n_seizures+1):
                last_sequence_end = max(filename_seizure_y.index[filename_seizure_y['pre1'] == i-1]) + 1
                current_sequence_start = last_sequence_end + config.POSTICTAL_SECONDS
                current_sequence_end = min(filename_seizure_y.index[filename_seizure_y['pre1'] == i])
                if current_sequence_end - current_sequence_start >= config.PREICTAL_SECONDS:
                    patient_idxs[patient].append(np.arange(current_sequence_start, current_sequence_end))

    seizure_x = load_paper_data(seizure_x_filepath, lazy=True)
    for patient, idxs in tqdm(patient_idxs.items(), desc="Seizure data: Patients"):
        n_seizures = 0
        for idx in tqdm(idxs, desc=f"{patient}", leave=False):
            with h5py.File(h5_filepath, 'a') as f:
                f.create_dataset(f"{patient}/test/X/{n_seizures}", data=seizure_x[idx])
            seizure_y.iloc[idx].to_hdf(str(h5_filepath), key=f"{patient}/test/y/{n_seizures}", mode="a")
            # f.create_dataset(f"{patient}/train/y/{n_normals}", data=normal_y.iloc[idx])
            n_seizures += 1


def main():
    warnings.filterwarnings('ignore', category=NaturalNameWarning)

    output_path = config.H5_DIRPATH
    h5_filepath = config.H5_FILEPATH
    if h5_filepath.exists():
        raise FileExistsError(f"{h5_filepath} already exists. Delete it first.")

    paper_dirpath = config.NUMPY_DATASET_PATH
    normal_x_filename = 'normal_1_0_data_x.npy'
    normal_y_filename = "normal_1_0_data_y.npy"
    seizure_x_filename = 'seizure_1_0_data_x.npy'
    seizure_y_filename = "seizure_1_0_data_y.npy"

    output_path.mkdir(exist_ok=True, parents=True)
    create_normal_datasets(h5_filepath, paper_dirpath/normal_x_filename, paper_dirpath/normal_y_filename)
    create_seizure_datasets(h5_filepath, paper_dirpath/seizure_x_filename, paper_dirpath/seizure_y_filename)


if __name__ == '__main__':
    main()
