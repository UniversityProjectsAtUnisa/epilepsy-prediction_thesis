import glob
import mne
import pathlib
import config
import numpy as np
import re
import json


def extract_data(edf_path: pathlib.Path) -> np.ndarray:
    raw_edf = mne.io.read_raw_edf(edf_path, stim_channel="")
    assert raw_edf.info['sfreq'] == config.SAMPLING_FREQUENCY  # type: ignore
    data = raw_edf.get_data()  # type: ignore
    return data.astype(np.float32) * 1e6  # type: ignore


def extract_slices(edf_path: pathlib.Path, summary: str):
    if edf_path.name not in summary:
        return []
    i_text_start = summary.index(edf_path.name)

    if 'File Name' in summary[i_text_start:]:
        i_text_stop = summary.index('File Name', i_text_start)
    else:
        i_text_stop = len(summary)
    assert i_text_stop > i_text_start

    file_text = summary[i_text_start:i_text_stop]

    m = re.search(r"Number of Seizures in File: ([0-9]*)", file_text)
    if m is None or m.group(1) == '0':
        useful_slices = [(0, extract_data(edf_path).shape[1] // config.SAMPLING_FREQUENCY, False)]
        return useful_slices
    n_seizures = int(m.group(1))

    useful_slices = []

    end_sec = 0
    for i in range(n_seizures):
        start_sec = int(re.search(f"Seizure(?: {i+1})? Start Time:\s*([0-9]*) seconds", file_text).group(1))  # type: ignore

        if i == 0:
            useful_slices.append((end_sec, start_sec, True))
        elif start_sec - end_sec > config.PREICTAL_SECONDS + config.POSTICTAL_SECONDS:
            useful_slices.append((end_sec + config.POSTICTAL_SECONDS, start_sec, True))

        end_sec = int(re.search(f"Seizure(?: {i+1})? End Time:\s*([0-9]*) seconds", file_text).group(1))  # type: ignore
        assert end_sec > start_sec

    X = extract_data(edf_path)
    end_sample = X.shape[1] // config.SAMPLING_FREQUENCY

    if end_sec + config.POSTICTAL_SECONDS < end_sample:
        useful_slices.append((end_sec+config.POSTICTAL_SECONDS, end_sample, False))

    return useful_slices


def load_data(subject_path: pathlib.Path):
    subject_id = subject_path.name
    edf_filenames = sorted(glob.glob(str(subject_path.joinpath("*.edf"))))
    summary = subject_path.joinpath(f"{subject_id}-summary.txt").read_text()
    if subject_id in config.PARTIAL_PATHNAMES:
        channel_configurations = summary.split("Channels")
        channels = config.PARTIAL_PATHNAMES[subject_id]
        summary = "".join(channel_configurations[i] for i in channels)

    slices = {}
    for edf_filename in edf_filenames:
        edf_path = pathlib.Path(edf_filename)
        local_slices = extract_slices(edf_path, summary)
        slices[edf_path.name] = local_slices
    return slices


def main():
    dataset_path = config.DATASET_PATH
    output_path = config.OUTPUT_PATH.joinpath(str(config.PREICTAL_SECONDS))
    output_path.mkdir(exist_ok=True, parents=True)
    if list(output_path.iterdir()):
        raise RuntimeError("Output path not empty")

    total_slices = {}
    for path in dataset_path.iterdir():
        if path.is_dir() and path.name not in config.FORBIDDEN_PATHNAMES:
            slices = load_data(path)
            total_slices[path.name] = slices

    with open(output_path.joinpath("slices.json"), "w") as f:
        json.dump(total_slices, f)


if __name__ == '__main__':
    main()
