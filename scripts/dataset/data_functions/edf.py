import mne
from typing import List
import pathlib
import numpy as np


def read_raw_edf(edf_path: pathlib.Path, useful_channels: List[str]) -> mne.io.Raw:
    raw_edf = mne.io.read_raw_edf(edf_path, stim_channel=None, include=useful_channels)  # type: ignore

    if "T8-P8-1" in raw_edf.ch_names:
        raw_edf.drop_channels("T8-P8-1")
        new_names = {"T8-P8-0": "T8-P8"}
        raw_edf.rename_channels(new_names)

    raw_edf.reorder_channels(useful_channels)
    return raw_edf  # type: ignore


def split_epochs(raw_edf: mne.io.Raw):
    return mne.make_fixed_length_epochs(raw_edf)  # type: ignore


def preprocess_epochs(epochs: mne.Epochs) -> mne.Epochs:
    epochs = epochs.load_data()
    return epochs  # type: ignore


def epochs_to_numpy(epochs: mne.Epochs):
    numpy_data = epochs.get_data().astype(np.float32)
    return numpy_data


def preprocess_numpy(numpy_data: np.ndarray):
    return numpy_data[:, ::2] * 1e6
