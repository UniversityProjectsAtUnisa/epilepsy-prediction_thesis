import config
import h5py
import json
import pathlib
import mne
import numpy as np


def other_files(output_path, filter_out):
    return list(filter(lambda x: x.name != filter_out, output_path.iterdir()))


def load_slices_metadata(output_path) -> dict[str, dict[str, list[list]]]:
    with open(output_path.joinpath(config.SLICES_FILENAME)) as f:
        return json.load(f)


def extract_data(edf_path: pathlib.Path) -> np.ndarray:
    # raw_edf = mne.io.read_raw_edf(edf_path, include=config.USEFUL_CHANNELS, stim_channel="")
    raw_edf = mne.io.read_raw_edf(edf_path, preload=True)
    print(edf_path)
    return
    print(raw_edf.ch_names)
    print("Missing:")
    for channel in config.USEFUL_CHANNELS:
        if channel not in raw_edf.ch_names:
            print(channel)
    print("extra")
    for channel in raw_edf.ch_names:
        if channel not in config.USEFUL_CHANNELS:
            print(channel)
    assert raw_edf.info['sfreq'] == config.SAMPLING_FREQUENCY  # type: ignore
    data = raw_edf.get_data()  # type: ignore
    return data.astype(np.float32) * 1e6  # type: ignore


def load_data(edf_path: pathlib.Path, segments) -> tuple[np.ndarray, np.ndarray]:
    data = extract_data(edf_path)
    assert data.shape[0] == len(config.USEFUL_CHANNELS)
    print(data.shape)
    return
    X = []
    for start, end, contains_seizure in segments:
        X.append(data[:, start * config.SAMPLING_FREQUENCY:end * config.SAMPLING_FREQUENCY])
    X_array = np.concatenate(X)


def main():
    dataset_path = config.DATASET_PATH
    output_path = config.OUTPUT_PATH.joinpath(str(config.PREICTAL_SECONDS))
    output_path.mkdir(exist_ok=True, parents=True)
    if other_files(output_path, config.SLICES_FILENAME):
        print(f"Output path contains files other than {config.SLICES_FILENAME}")
        return

    slices_metadata = load_slices_metadata(output_path)
    for patient, patient_slices in slices_metadata.items():
        for edf_filename, segments in patient_slices.items():
            edf_path = dataset_path.joinpath(patient, edf_filename)
            X, y = load_data(edf_path, segments)
            return
            with h5py.File(output_path.joinpath(config.H5_FILENAME), "a") as f:
                f.create_dataset(f"{patient}/X", data=X)
                f.create_dataset(f"{patient}/y", data=y)


if __name__ == '__main__':
    main()
