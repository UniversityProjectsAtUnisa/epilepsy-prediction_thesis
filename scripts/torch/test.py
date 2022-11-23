from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector
import torch
from typing import List, Tuple
from utils.gpu_utils import device_context
import numpy as np


def get_time_left(windows_left: int):
    return windows_left * (config.WINDOW_SIZE_SECONDS - config.WINDOW_OVERLAP_SECONDS)


def consecutive_preds(pred, consecutive_windows):
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be >= 1")
    if len(pred) < consecutive_windows:
        raise ValueError("Not enough windows to evaluate")
    if consecutive_windows == 1:
        return pred
    ps = []
    for i in range(len(pred) - (consecutive_windows - 1)):
        ps.append(pred[i:i + consecutive_windows+1].sum() >= consecutive_windows/2)
    return np.array(ps)


def print_sample_evaluations(preds: Tuple[np.ndarray], consecutive_windows=1):
    correct_predictions = 0
    average_time_left = 0
    not_found = 0
    preds = tuple(consecutive_preds(pred, consecutive_windows) for pred in preds)
    for sample_pred in preds:
        occurrence_indices = np.flatnonzero(sample_pred == 1)
        if len(occurrence_indices) == 0:
            not_found += 1
            continue
        first_occurrence_index = int(occurrence_indices[0])
        time_left = get_time_left(len(sample_pred) - first_occurrence_index)
        correct_predictions += time_left <= config.PREICTAL_SECONDS
        average_time_left += time_left
    if len(preds) == not_found:
        print("No predictions found")
        return
    average_time_left /= len(preds) - not_found
    print(f"Accuracy: {correct_predictions / len(preds):.3f} ({correct_predictions} / {len(preds)})")
    print(f"Samples not found: {not_found} / {len(preds)}")
    print(f"Average time left: {average_time_left} seconds")


def evaluate(preds: Tuple[np.ndarray]):
    for i in range(1, 10):
        print(f"Consecutive windows: {i}")
        print_sample_evaluations(preds, i)
        print()


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    _, X_test = load_data(config.H5_PATH.joinpath(config.H5_FILENAME), "chb15", load_train=False)

    if X_test is None:
        raise ValueError("No test data found")

    X_test = convert_to_tensor(*X_test)

    with device_context:
        model = AnomalyDetector.load(dirpath)
        preds = tuple(model.predict(x).cpu().numpy() for x in X_test)

    evaluate(preds)


if __name__ == '__main__':
    main()
