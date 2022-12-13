import glob
import mne
import pathlib
import config
import numpy as np
import re
import pandas as pd
from typing import Tuple


def extract_data(edf_path: pathlib.Path) -> np.ndarray:
    raw_edf = mne.io.read_raw_edf(edf_path, stim_channel="")
    assert raw_edf.info['sfreq'] == config.SAMPLING_FREQUENCY  # type: ignore
    data = raw_edf.get_data()  # type: ignore
    return data.astype(np.float32) * 1e6  # type: ignore


def load_paper_labels(filepath: pathlib.Path):
    data_y = np.load(filepath, allow_pickle=True)  # metadatos

    data_y = pd.DataFrame(data_y, columns=['type', 'name', 'filename', 'pre1', 'pre2', 'id_eeg_actual', 'id_eeg_all', 'label'])
    data_y.label = data_y.label.astype('int')

    return data_y


def contains_seizure(edf_filepath: pathlib.Path) -> bool:
    return edf_filepath.with_suffix('.edf.seizures').exists()


def load_duration(edf_filepath: pathlib.Path) -> Tuple[int, bool]:
    return extract_data(edf_filepath).shape[1] // config.SAMPLING_FREQUENCY


def count_total_files(dataset_path: pathlib.Path) -> Tuple[int, int]:
    total_normal_files = 0
    total_seizure_files = 0
    for dirpath in dataset_path.iterdir():
        if dirpath.is_dir():
            for filepath in glob.glob(str(dirpath/"*.edf")):
                if not contains_seizure(pathlib.Path(filepath)):
                    total_normal_files += 1
                else:
                    total_seizure_files += 1
    return total_normal_files, total_seizure_files


def main():
    dataset_path = config.DATASET_PATH
    paper_dirpath = pathlib.Path('/media/HDD/Unisa/Datasets/EEG data/')
    normal_y_filename = "normal_1_0_data_y.npy"
    seizure_y_filename = "seizure_1_0_data_y.npy"

    paper_y_normal = load_paper_labels(paper_dirpath/normal_y_filename)
    paper_y_seizure = load_paper_labels(paper_dirpath/seizure_y_filename)

    paper_normal_filenames = set(paper_y_normal['filename'].unique())
    paper_seizure_filenames = set(paper_y_seizure['filename'].unique())
    paper_filenames = paper_normal_filenames.union(paper_seizure_filenames)

    total_normal_files, total_seizure_files = count_total_files(dataset_path)

    seizure_missing = [f'in paper labels: {len(paper_seizure_filenames)}/{total_seizure_files}']
    normal_missing = [f'in paper labels: {len(paper_normal_filenames)}/{total_normal_files}']
    missing = [f'in paper labels: {len(paper_filenames)}/{total_normal_files + total_seizure_files}']
    mismatch = []

    for path in dataset_path.iterdir():
        if path.is_dir():
            edf_filenames = sorted(glob.glob(str(path.joinpath("*.edf"))))
            for edf_filename in edf_filenames:
                edf_filepath = pathlib.Path(edf_filename)
                has_seizures = contains_seizure(edf_filepath)
                if edf_filepath.name not in paper_filenames:
                    if has_seizures:
                        seizure_missing.append(edf_filepath.name)
                    else:
                        normal_missing.append(edf_filepath.name)
                    missing.append(edf_filepath.name)
                    continue
                duration = load_duration(edf_filepath)
                if has_seizures:
                    n_samples = len(paper_y_seizure[paper_y_seizure['filename'] == edf_filepath.name])
                else:
                    n_samples = len(paper_y_normal[paper_y_normal['filename'] == edf_filepath.name])
                if n_samples != duration:
                    mismatch.append(f"{edf_filename}: {duration}; paperdata: {n_samples}")

    pathlib.Path('missing.txt').write_text('\n'.join(missing))
    pathlib.Path('seizure_missing.txt').write_text('\n'.join(seizure_missing))
    pathlib.Path('normal_missing.txt').write_text('\n'.join(normal_missing))
    pathlib.Path('mismatch.txt').write_text('\n'.join(mismatch))


if __name__ == '__main__':
    main()
