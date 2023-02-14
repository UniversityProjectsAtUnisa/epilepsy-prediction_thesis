from pathlib import Path
from ..data_functions import preprocess_data, segment_data, convert_to_tensor
from .model.cnn_anomaly_detector import CNNAnomalyDetector
from scripts.dataset.data_functions.load import load_paper_labels
import h5py
import numpy as np


def load_x_data(dataset_path, key):
    with h5py.File(dataset_path) as f:
        x = f[key][:]  # type: ignore
    return x  # type: ignore


def main():
    model_path = Path("/media/marco741/Archive/Datasets/spectrograms/patientgeneric_balancedvalloss_nocheckstop/chb12/fold_3")
    h5_filepath = Path("/media/marco741/Archive/Datasets/spectrograms/dataset.h5")

    normal_file_keys = [
        "chb12/train/X/0",
        "chb12/train/X/1",
    ]
    seizure_file_keys = [
        "chb12/test/X/0",
        "chb12/test/X/2",
    ]

    model = CNNAnomalyDetector.load(model_path, svm_dirpath=model_path/"i_4")

    data = []
    for fkey in normal_file_keys:
        x = load_x_data(h5_filepath, fkey)[:600]
        x = segment_data(x, 6, 0)
        data.append(x)

    for fkey in seizure_file_keys:
        x = load_x_data(h5_filepath, fkey)
        x = segment_data(x, 6, 0)
        data.append(x)

    data = tuple(data)
    data = convert_to_tensor(*data)

    preds = tuple(np.argwhere(model.predict(x).numpy() > 0).flatten() for x in data)

    print(preds)
    print()


if __name__ == '__main__':
    main()
