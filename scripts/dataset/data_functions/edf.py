from scipy import signal
import numpy as np
import pathlib
from typing import List
import mne
import matplotlib.pyplot as plt


class EDFOperations:
    def __init__(self, use_spectrograms: bool = False):
        self.use_spectrograms = use_spectrograms

    def read_raw_edf(self, edf_path: pathlib.Path, useful_channels: List[str]) -> mne.io.Raw:
        raw_edf = mne.io.read_raw_edf(edf_path, stim_channel=None, include=useful_channels)  # type: ignore

        if "T8-P8-1" in raw_edf.ch_names:
            raw_edf.drop_channels("T8-P8-1")
            new_names = {"T8-P8-0": "T8-P8"}
            raw_edf.rename_channels(new_names)

        raw_edf.reorder_channels(useful_channels)
        return raw_edf  # type: ignore

    def split_epochs(self, raw_edf: mne.io.Raw):
        return mne.make_fixed_length_epochs(raw_edf)  # type: ignore

    def preprocess_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        epochs = epochs.load_data()
        if self.use_spectrograms:
            epochs = epochs.filter(8, 13)  # type: ignore
        return epochs  # type: ignore

    def epochs_to_numpy(self, epochs: mne.Epochs):
        numpy_data = epochs.get_data().astype(np.float32)
        return numpy_data

    def preprocess_numpy(self, data: np.ndarray):
        if self.use_spectrograms:
            prepend = np.expand_dims(np.concatenate([np.array([data[0, :, 0]]), data[:-1, :, -1]]), -1)
            data = abs(np.diff(data, prepend=prepend))
            data = self.to_spectrogram(data)
            return data.squeeze()
        return data[:, :, ::2] * 1e6

    def to_spectrogram(self, data: np.ndarray):
        _, _, Pxx = signal.spectrogram(data, fs=256, noverlap=0)
        Pxx = Pxx[..., :40, :]  # type: ignore
        Pxx[Pxx == 0] = Pxx[Pxx != 0].min()
        Pxx = 10*np.log10(Pxx)
        return Pxx


def plot_spectrogram(ch_data):
    plt.figure()
    plt.pcolormesh(np.arange(ch_data.shape[1])+0.5, np.arange(ch_data.shape[0]), ch_data, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()
