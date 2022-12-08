import copy
import json
import pathlib
from typing import Dict, List, Tuple

import config
import h5py
import mne
import numpy as np
from tqdm import tqdm
from scipy import signal
import cv2
from utils import ResizableH5Dataset
from feature_extraction import featureExtraction


def other_files(output_path, *filter_out):
    return list(filter(lambda x: x.name not in filter_out, output_path.iterdir()))


def load_slices_metadata(output_path) -> Dict[str, Dict[str, Dict[str, List[List]]]]:
    with open(output_path.joinpath(config.SLICES_FILENAME)) as f:
        return json.load(f)


def read_raw_edf(edf_path: pathlib.Path) -> mne.io.Raw:
    raw_edf = mne.io.read_raw_edf(edf_path, stim_channel=None, include=config.USEFUL_CHANNELS)  # type: ignore

    if "T8-P8-1" in raw_edf.ch_names:
        raw_edf.drop_channels("T8-P8-1")
        new_names = {"T8-P8-0": "T8-P8"}
        raw_edf.rename_channels(new_names)

    raw_edf.reorder_channels(config.USEFUL_CHANNELS)
    return raw_edf  # type: ignore


def to_spectrogram(data: np.ndarray):
    new_data = []
    for window in data:
        new_window = []
        for channel in window:
            _, _, Pxx = signal.spectrogram(channel, nfft=256, fs=256, noverlap=0)
            spect = cv2.flip(np.uint8(10*np.log10(Pxx)), 0)  # type: ignore
            new_window.append(spect[-40:])

        new_data.append(np.array(new_window))
    return np.array(new_data)


def extract_data(raw: mne.io.Raw, use_spectrograms: bool) -> np.ndarray:
    raw = raw.load_data()
    data = raw.get_data().astype(np.float32) * 1e6  # type: ignore

    data = np.array([featureExtraction(x, config.SAMPLING_FREQUENCY, exp="AVERAGE") for x in data])

    # data = raw.filter(8, 13).get_data().astype(np.float32) * 1e6  # type: ignore
    # data = abs(np.diff(data, prepend=data[:, :, 0].reshape(data.shape[0], data.shape[1], 1)))
    # if use_spectrograms:
    #     data = to_spectrogram(data)
    return data


def split_epochs(edf: mne.io.Raw, duration: float, overlap: float) -> mne.io.Raw:
    return mne.make_fixed_length_epochs(edf, duration=duration, overlap=overlap)  # type: ignore


def adjust_training_segment_duration(segment: List, duration: float, overlap: float):
    start, end, _ = segment
    segment_duration = end - start
    effective_window_size = duration - overlap
    skip = (segment_duration - overlap) % effective_window_size
    segment = copy.deepcopy(segment)
    segment[0] += skip
    return segment


def extract_test_data(edf_path: pathlib.Path, segments: List[List],
                      duration: float, overlap: float, use_spectrograms: bool = False) -> Tuple[List[np.ndarray],
                                                                                                List[np.ndarray]]:
    normal = []
    ictal = []
    for segment in segments:
        segment = adjust_training_segment_duration(segment, duration, overlap)
        data = extract_single_segment_data(edf_path, segment, duration, overlap, use_spectrograms)
        if segment[2]:
            ictal.append(data)
        else:
            normal.append(data)
    return normal, ictal


def extract_training_data(edf_path: pathlib.Path, segments: List[List], duration: float, overlap: float, use_spectrograms: bool = False) -> np.ndarray:
    assert len(segments) == 1 and segments[0][2] == False
    return extract_single_segment_data(edf_path, segments[0], duration, overlap, use_spectrograms)


def extract_single_segment_data(edf_path: pathlib.Path, segment: List, duration: float, overlap: float, use_spectrograms: bool) -> np.ndarray:
    start, end, _ = segment
    raw_edf = read_raw_edf(edf_path)
    raw_edf = raw_edf.copy().crop(start, end, include_tmax=False)
    epochs = split_epochs(raw_edf, duration, overlap)
    return extract_data(epochs, use_spectrograms)


def main():
    mne.set_log_level("ERROR")
    dataset_path = config.DATASET_PATH
    output_path = config.H5_DIRPATH
    output_path.mkdir(exist_ok=True, parents=True)
    if other_files(output_path, config.SLICES_FILENAME, config.SLICES_ANALYSIS_FILENAME):
        print(f"Output path contains files other than {config.SLICES_FILENAME}")
        return

    slices_metadata = load_slices_metadata(output_path)
    for patient, patient_slices in tqdm(slices_metadata.items(), desc="Patients"):
        if patient != "chb15":
            continue
        n_normals = 0
        n_files = 0

        with h5py.File(output_path.joinpath(config.H5_FILENAME), "a") as f:
            normal_test_dataset = ResizableH5Dataset(f"{patient}/test/normal")
            ictal_test_dataset = ResizableH5Dataset(f"{patient}/test/ictal")
            for edf_filename, content in tqdm(patient_slices.items(), desc=f"{patient}", leave=False):
                if edf_filename in config.DISCARDED_EDFS:
                    continue
                n_seizures = content["n_seizures"]
                slices = content["slices"]
                edf_path = dataset_path.joinpath(patient, edf_filename)
                if n_seizures == 0:
                    normal = extract_training_data(edf_path, slices, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS, config.USE_SPECTROGRAMS)
                    f.create_dataset(f"{patient}/train/{n_normals}", data=normal)
                    n_normals += 1
                else:
                    normal, ictal = extract_test_data(edf_path, slices, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS, config.USE_SPECTROGRAMS)
                    for test in normal:
                        normal_test_dataset.append_data(f, data=test)
                    for test in ictal:
                        ictal_test_dataset.append_data(f, data=test)
                    n_files += 1


if __name__ == '__main__':
    main()
