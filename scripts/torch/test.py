from data_functions import load_data, convert_to_tensor, load_patient_names, split_data
from evaluation import quality_metrics as qm
from evaluation import plot_functions as pf
from model.helpers.history import History
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from model.autoencoder import Autoencoder
from utils.gpu_utils import device_context

import numpy as np
import math
import pandas as pd


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


def print_sample_evaluations(preds, consecutive_windows=1):
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


def evaluate(preds, preictal_seconds, window_size_seconds, window_overlap_seconds):
    n_positive_windows = math.ceil(preictal_seconds // (window_size_seconds - window_overlap_seconds))
    metrics = {}
    metrics['seizures'] = len(preds)
    metrics['undetected%'] = 100*qm.undetected_predictions(preds)
    metrics['pred%'] = 100*qm.prediction_accuracy(preds, n_positive_windows)
    metrics['spec%'] = 100*qm.specificity(preds, n_positive_windows)
    metrics['APT (s)'], metrics['mPT (s)'], metrics['MPT (s)'] = qm.prediction_seconds_before_seizure(preds, window_size_seconds, window_overlap_seconds)
    metrics['PPV (%)'] = 100*qm.ppv(preds, n_positive_windows)
    metrics['BPPV (%)'] = 100*qm.bppv(preds, n_positive_windows)
    metrics['Imb. (%)'] = 100*qm.imbalanceness(preds, n_positive_windows)
    return metrics


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = sorted(load_patient_names(config.H5_FILEPATH))

    percentiles = np.linspace(95, 100, 11, endpoint=True)
    metrics_df = {perc: pd.DataFrame(columns=config.METRIC_NAMES) for perc in percentiles}

    for patient_name in patient_names:
        print(f"Testing for patient {patient_name}")
        patient_dirpath = dirpath.joinpath(patient_name)
        X_normal, X_test = load_data(config.H5_FILEPATH, patient_name)
        if not X_normal:
            raise ValueError("No training data found")
        _, X_val = split_data(X_normal, random_state=config.RANDOM_STATE)

        if X_test is None:
            raise ValueError("No test data found")

        X_val, = convert_to_tensor(X_val)
        X_test = convert_to_tensor(*X_test)

        history = History.load(patient_dirpath/AnomalyDetector.model_dirname/Autoencoder.history_filename)
        history_plot = pf.plot_train_val_losses(history.train, history.val)
        history_plot.savefig(str(patient_dirpath/config.LOSS_PLOT_FILENAME))

        with device_context:
            model = AnomalyDetector.load(patient_dirpath)
            for perc in percentiles:
                losses_val = model.model.calculate_losses(X_val)  # type: ignore
                model.threshold.threshold = np.percentile(losses_val, perc)  # type: ignore
                preds = tuple(model.predict(x) for x in X_test)

                metrics_dict = evaluate(preds, config.PREICTAL_SECONDS, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
                new_metrics_df = pd.DataFrame([metrics_dict], index=[patient_name])
                metrics_df[perc] = pd.concat([metrics_df[perc], new_metrics_df])
        print()

    for k, m in metrics_df.items():
        m.to_csv(dirpath.joinpath(f"{k}_{config.METRICS_FILENAME}"))


if __name__ == '__main__':
    main()
