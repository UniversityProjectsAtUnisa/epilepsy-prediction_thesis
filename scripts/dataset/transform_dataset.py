import glob
import code
import os
import mne
import pathlib
import config
import numpy as np


def extract_data():
    pass


def load_data(subject_path: pathlib.Path):
    subject_id = subject_path.name
    edf_filenames = sorted(glob.glob(str(subject_path.joinpath("*.edf"))))
    summary = subject_path.joinpath(f"{subject_id}-summary.txt").read_text()
    for edf_filename in edf_filenames:
        raw_edf = mne.io.read_raw_edf(pathlib.Path(edf_filename).name, stim_channel="")
        raw_data = raw_edf.get_data()
        # if raw_data
        raw_edf.get_data().astype(np.float32) * 1e6  # to mV

    code.interact(local=dict(globals(), **locals()))

    # return


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
