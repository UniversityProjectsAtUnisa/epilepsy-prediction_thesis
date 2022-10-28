import glob
import code
import os
import mne
import pathlib
import config
import numpy as np
import re


def extract_data(edf_path: pathlib.Path) -> np.ndarray:
    raw_edf = mne.io.read_raw_edf(edf_path, stim_channel="")
    assert raw_edf.info['sfreq'] == config.SAMPLING_FREQUENCY  # type: ignore
    data = raw_edf.get_data()  # type: ignore
    # code.interact(local=dict(globals(), **locals()))
    return data.astype(np.float32) * 1e6  # type: ignore


def extract_labels(summary: str, edf_path: pathlib.Path, size: int) -> np.ndarray:
    labels = np.zeros(size, dtype=np.int32)
    i_text_start = summary.index(edf_path.name)

    if 'File Name' in summary[i_text_start:]:
        i_text_stop = summary.index('File Name', i_text_start)
    else:
        i_text_stop = len(summary)
    assert i_text_stop > i_text_start

    file_text = summary[i_text_start:i_text_stop]
    if 'Seizure Start' in file_text:
        start_sec = int(re.search(r"Seizure Start Time: ([0-9]*) seconds", file_text).group(1))  # type: ignore
        end_sec = int(re.search(r"Seizure End Time: ([0-9]*) seconds", file_text).group(1))  # type: ignore
        i_seizure_start = int(round(start_sec * config.SAMPLING_FREQUENCY))
        i_seizure_stop = int(round((end_sec + 1) * config.SAMPLING_FREQUENCY))
        labels[i_seizure_start:i_seizure_stop] = 1

    return labels


def extract_data_and_labels(edf_filename: str, summary: str):
    edf_path = pathlib.Path(edf_filename)

    i_text_start = summary.index(edf_path.name)

    if 'File Name' in summary[i_text_start:]:
        i_text_stop = summary.index('File Name', i_text_start)
    else:
        i_text_stop = len(summary)
    assert i_text_stop > i_text_start

    file_text = summary[i_text_start:i_text_stop]

    m = re.search(r"Number of Seizures in File: ([0-9]*)", file_text)
    if m is None or m.group(1) == '0':
        useful_slices = [(0, extract_data(edf_path).shape[1] // config.SAMPLING_FREQUENCY)]
        print(useful_slices)
        return useful_slices
    n_seizures = int(m.group(1))

    useful_slices = []

    end_sec = 0
    for i in range(n_seizures):
        start_sec = int(re.search(f"Seizure {i+1} Start Time: ([0-9]*) seconds", file_text).group(1))  # type: ignore

        # if i == 0 and start_sec - end_sec > config.PREICTAL_SECONDS:
        if i == 0:
            useful_slices.append((end_sec, start_sec))
        elif start_sec - end_sec > config.PREICTAL_SECONDS + config.POSTICTAL_SECONDS:
            useful_slices.append((end_sec + config.POSTICTAL_SECONDS, start_sec))

        end_sec = int(re.search(f"Seizure {i+1} End Time: ([0-9]*) seconds", file_text).group(1))  # type: ignore
        assert end_sec > start_sec

    X = extract_data(edf_path)
    end_sample = X.shape[1] // config.SAMPLING_FREQUENCY

    if end_sec + config.POSTICTAL_SECONDS < end_sample:
        useful_slices.append((end_sec+config.POSTICTAL_SECONDS, end_sample))

    print(useful_slices)
    return useful_slices

    # print(f"Seizure {i} Start Time: {start_sec} seconds End Time: {end_sec} seconds")

    # if 'Seizure Start' not in file_text:
    #     X = extract_data(edf_path)
    #     y = np.zeros(X.shape[1], dtype=np.int32)
    #     return X, y

    #     start_sec = int(re.search(r"Seizure Start Time: ([0-9]*) seconds", file_text).group(1))  # type: ignore
    #     end_sec = int(re.search(r"Seizure End Time: ([0-9]*) seconds", file_text).group(1))  # type: ignore
    #     i_seizure_start = int(round(start_sec * config.SAMPLING_FREQUENCY))
    #     i_seizure_stop = int(round((end_sec + 1) * config.SAMPLING_FREQUENCY))
    #     labels[i_seizure_start:i_seizure_stop] = 1


def load_data(subject_path: pathlib.Path):
    subject_id = subject_path.name
    edf_filenames = sorted(glob.glob(str(subject_path.joinpath("*.edf"))))
    summary = subject_path.joinpath(f"{subject_id}-summary.txt").read_text()
    for edf_filename in edf_filenames:
        extract_data_and_labels(edf_filename, summary)
        # y = extract_labels(summary, edf_path, X.shape[1])
        # X = extract_data(edf_path)
        # y = extract_labels(summary, edf_path, X.shape[1])
        # assert X.shape[1] == len(y)


def main():
    dataset_path = config.DATASET_PATH
    for path in dataset_path.iterdir():
        if path.is_dir():
            _ = load_data(path)

    edf = os.path.join(dataset_path, 'chb01', 'chb01_03.edf')
    sf = os.path.join(dataset_path, 'chb01', 'chb01_04.edf.seizures')
    raw = mne.io.read_raw_edf(edf)
    bytes = pathlib.Path(sf).read_bytes()

    # for encoding in encodings:
    #     try:
    #         text = bytes.decode(encoding)
    #         print(text, end=' >>> ')
    #         print(encoding)
    #     except:
    #         pass

    # print(raw.info)


if __name__ == '__main__':
    main()
