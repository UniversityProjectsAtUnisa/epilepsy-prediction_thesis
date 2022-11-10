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


def read_raw_edf(edf_path: pathlib.Path) -> mne.io.Raw:
    raw_edf = mne.io.read_raw_edf(edf_path, stim_channel=None, include=config.USEFUL_CHANNELS, verbose="ERROR")  # type: ignore

    if "T8-P8-1" in raw_edf.ch_names:
        raw_edf.drop_channels("T8-P8-1")
        new_names = {"T8-P8-0": "T8-P8"}
        raw_edf.rename_channels(new_names)

    raw_edf.reorder_channels(config.USEFUL_CHANNELS)
    return raw_edf  # type: ignore


def extract_data(raw: mne.io.Raw) -> np.ndarray:
    raw = raw.load_data()
    data = raw.filter(8, 13).get_data().as_type(np.float32) * 1e6  # type: ignore
    return data


def split_ictals(raw_edf: mne.io.Raw, segments: list) -> tuple[list[mne.io.Raw], list[mne.io.Raw]]:
    interictals = []
    preictals = []
    for start, end, contains_seizure in segments:
        if contains_seizure:
            preictal_start = min(start, end - config.PREICTAL_SECONDS)
            preictals.append(raw_edf.crop(preictal_start, end))
            if preictal_start > start:
                interictals.append(raw_edf.crop(start, preictal_start))
        else:
            interictals.append(raw_edf.crop(start, end))
    return interictals, preictals


def split_epochs(edfs: list[mne.io.Raw], duration: int, overlap: int) -> list[mne.io.Raw]:
    epochs_groups = []
    for edf in edfs:
        epochs = mne.make_fixed_length_epochs(edf, duration=duration, overlap=overlap)
        epochs_groups.append(epochs)
    return epochs_groups


def extract_data_and_labels(edf_path: pathlib.Path, segments: list) -> tuple[list[np.ndarray], list[np.ndarray]]:
    raw_edf = read_raw_edf(edf_path)
    interictals, preictals = split_ictals(raw_edf, segments)

    interictal_epochs_groups = split_epochs(interictals, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
    interictal_data = []
    for interictal_epochs in interictal_epochs_groups:
        interictal_data.append(extract_data(interictal_epochs))

    preictal_epochs_groups = split_epochs(preictals, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
    preictal_data = []
    for interictal_epochs in preictal_epochs_groups:
        preictal_data.append(extract_data(interictal_epochs))

    return interictal_data, preictal_data


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
            if edf_filename in config.DISCARDED_EDFS:
                continue
            edf_path = dataset_path.joinpath(patient, edf_filename)
            X, y = load_data(edf_path, segments)
            return
            with h5py.File(output_path.joinpath(config.H5_FILENAME), "a") as f:
                f.create_dataset(f"{patient}/X", data=X)
                f.create_dataset(f"{patient}/y", data=y)


if __name__ == '__main__':
    main()
