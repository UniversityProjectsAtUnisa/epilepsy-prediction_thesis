from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector
import torch
from typing import List, Tuple
from utils.gpu_utils import device_context


def get_time_left(windows_left: int):
    return windows_left * (config.WINDOW_SIZE_SECONDS - config.WINDOW_OVERLAP_SECONDS)


def print_sample_evaluations(preds: Tuple[torch.Tensor]):
    correct_predictions = 0
    average_time_left = 0
    not_found = 0
    for sample_pred in preds:
        occurrence_indices = ((sample_pred == 1).nonzero(as_tuple=True)[0])
        if len(occurrence_indices) == 0:
            not_found += 1
            continue
        first_occurrence_index = int(occurrence_indices[0][0])
        time_left = get_time_left(len(sample_pred) - first_occurrence_index)
        correct_predictions += time_left <= config.PREICTAL_SECONDS
        average_time_left += time_left
    average_time_left /= len(preds) - not_found
    print(f"Accuracy: {correct_predictions / len(preds)} ({correct_predictions} / {len(preds)})")
    print(f"Average time left: {average_time_left} seconds")


def evaluate(preds: Tuple[torch.Tensor]):
    print_sample_evaluations(preds)


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
        preds = tuple(model.predict(x) for x in X_test)

    evaluate(preds)


if __name__ == '__main__':
    main()
