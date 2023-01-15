import pathlib
import warnings
import mne
import h5py
import numpy as np
from tqdm import tqdm
from typing import List
from tables import NaturalNameWarning
from pandas.errors import PerformanceWarning

import dataset_config as config
from data_functions.load import load_paper_labels
from data_functions.edf import read_raw_edf, split_epochs, preprocess_epochs, epochs_to_numpy, preprocess_numpy


def load_patient_data(data_filepath: pathlib.Path, useful_channels: List[str]) -> np.ndarray:
    raw = read_raw_edf(data_filepath, useful_channels)
    epochs = split_epochs(raw)
    preprocessed_epochs = preprocess_epochs(epochs)
    numpy_epochs_data = epochs_to_numpy(preprocessed_epochs)
    data = np.array(list(map(preprocess_numpy, numpy_epochs_data)))
    return data


def create_seizure_datasets(h5_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, seizure_labels_filepath: pathlib.Path, useful_channels: List[str]):
    seizure_y = load_paper_labels(seizure_labels_filepath)

    patient_names = seizure_y['name'].unique()
    for patient_name in tqdm(patient_names, desc="Seizure data: Patients"):
        patient_filenames = seizure_y[seizure_y["name"] == patient_name]['filename'].unique()
        n_patient_seizures = 0
        for filename in tqdm(patient_filenames, desc=f"{patient_name}", leave=False):
            file_labels = seizure_y[seizure_y['filename'] == filename]
            n_file_seizures = max(file_labels['pre1'])

            file_idxs = []
            current_sequence_start = min(file_labels.index[file_labels['pre1'] == 0])
            current_sequence_end = min(file_labels.index[file_labels['pre1'] == 1])
            if current_sequence_end - current_sequence_start >= config.PREICTAL_SECONDS:
                file_idxs.append(np.arange(current_sequence_start - file_labels.index[0], current_sequence_end - file_labels.index[0]))
            for i in range(2, n_file_seizures+1):
                last_sequence_end = max(file_labels.index[file_labels['pre1'] == i-1]) + 1
                current_sequence_start = last_sequence_end + config.POSTICTAL_SECONDS
                current_sequence_end = min(file_labels.index[file_labels['pre1'] == i])
                if current_sequence_end - current_sequence_start >= config.PREICTAL_SECONDS:
                    file_idxs.append(np.arange(current_sequence_start - file_labels.index[0], current_sequence_end - file_labels.index[0]))

            data = load_patient_data(dataset_dirpath/patient_name/filename, config.USEFUL_CHANNELS)
            for idxs in file_idxs:
                with h5py.File(h5_filepath, 'a') as f:
                    f.create_dataset(f"{patient_name}/test/X/{n_patient_seizures}", data=data[idxs])
                file_labels.iloc[idxs].to_hdf(str(h5_filepath), key=f"{patient_name}/test/y/{n_patient_seizures}", mode="a")
                n_patient_seizures += 1


def create_normal_datasets(h5_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, normal_labels_filepath: pathlib.Path, useful_channels: List[str]):
    normal_y = load_paper_labels(normal_labels_filepath)

    patient_names = normal_y['name'].unique()
    for patient_name in tqdm(patient_names, desc="Normal data: Patients"):
        n_normals = 0
        patient_filenames = normal_y[normal_y["name"] == patient_name]['filename'].unique()
        for filename in tqdm(patient_filenames, desc=f"{patient_name}", leave=False):
            data = load_patient_data(dataset_dirpath/patient_name/filename, useful_channels)
            labels = normal_y[normal_y['filename'] == filename]
            with h5py.File(h5_filepath, 'a') as f:
                f.create_dataset(f"{patient_name}/train/X/{n_normals}", data=data)
            labels.to_hdf(str(h5_filepath), key=f"{patient_name}/train/y/{n_normals}", mode="a")
            n_normals += 1


def main():
    mne.set_log_level("ERROR")
    warnings.filterwarnings('ignore', category=NaturalNameWarning)
    warnings.filterwarnings('ignore', category=PerformanceWarning)

    h5_filepath = config.H5_FILEPATH
    if h5_filepath.exists():
        # raise FileExistsError(f"{h5_filepath} already exists. Delete it first.")
        pass
    h5_filepath.parent.mkdir(exist_ok=True, parents=True)

    dataset_dirpath = config.DATASET_DIRPATH
    normal_labels_filepath = config.NORMAL_LABELS_FILEPATH
    seizure_labels_filepath = config.SEIZURE_LABELS_FILEPATH
    useful_channels = config.USEFUL_CHANNELS

    create_normal_datasets(h5_filepath, dataset_dirpath, normal_labels_filepath, useful_channels)
    create_seizure_datasets(h5_filepath, dataset_dirpath, seizure_labels_filepath, useful_channels)


if __name__ == '__main__':
    main()
