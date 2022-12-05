from data_functions import load_data, convert_to_tensor, load_patient_names, split_data
from evaluation import quality_metrics as qm
from evaluation import plot_functions as pf
from model.helpers.history import History
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from model.autoencoder import Autoencoder
from utils.gpu_utils import device_context
from skimage.filters.thresholding import threshold_otsu

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


def evaluate(normal_preds, ictal_preds):
    metrics = {}
    metrics["normal_accuracy"] = 100*float(sum(normal_preds)/len(normal_preds))
    metrics["ictal_accuracy"] = 100*float(sum(ictal_preds)/len(ictal_preds))
    metrics['Imb. (%)'] = 100*float(len(ictal_preds)/(len(normal_preds)+len(ictal_preds)))
    return metrics


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = sorted(load_patient_names(config.H5_FILEPATH))

    percentiles = np.linspace(95, 100, 11, endpoint=True)
    metrics_df = {perc: pd.DataFrame(columns=['normal_accuracy', 'ictal_accuracy', 'Imb. (%)']) for perc in percentiles}
    otsu_df = pd.DataFrame(columns=['normal_accuracy', 'ictal_accuracy', 'Imb. (%)'])

    for patient_name in patient_names:
        print(f"Testing for patient {patient_name}")
        patient_dirpath = dirpath.joinpath(patient_name)
        X_normal, X_test_normal, X_test_ictal = load_data(config.H5_FILEPATH, patient_name, preprocess=not config.USE_CONVOLUTION)
        if not X_normal:
            raise ValueError("No training data found")
        _, X_val = split_data(X_normal, random_state=config.RANDOM_STATE)

        if X_test_normal is None or X_test_ictal is None:
            raise ValueError("No test data found")

        X_val, X_test_normal, X_test_ictal = convert_to_tensor(X_val, X_test_normal, X_test_ictal)

        history = History.load(patient_dirpath/AnomalyDetector.model_dirname/Autoencoder.history_filename)
        history_plot = pf.plot_train_val_losses(history.train, history.val)
        history_plot.savefig(str(patient_dirpath/config.LOSS_PLOT_FILENAME))

        with device_context:
            model = AnomalyDetector.load(patient_dirpath)
            losses_val = model.model.calculate_losses(X_val)  # type: ignore
            for perc in percentiles:
                model.threshold.threshold = np.percentile(losses_val, perc)  # type: ignore
                normal_preds = model.predict(X_test_normal)
                ictal_preds = model.predict(X_test_ictal)

                metrics_dict = evaluate(normal_preds, ictal_preds)
                new_metrics_df = pd.DataFrame([metrics_dict], index=[patient_name])
                metrics_df[perc] = pd.concat([metrics_df[perc], new_metrics_df])
            # Otsu's method
            model.threshold.threshold = threshold_otsu(np.array(losses_val))  # type: ignore
            normal_preds = model.predict(X_test_normal)
            ictal_preds = model.predict(X_test_ictal)

            metrics_dict = evaluate(normal_preds, ictal_preds)
            new_metrics_df = pd.DataFrame([metrics_dict], index=[patient_name])
            otsu_df = pd.concat([otsu_df, new_metrics_df])

        print()

    for k, m in metrics_df.items():
        m.round(2).to_csv(dirpath.joinpath(f"{k}_{config.METRICS_FILENAME}"))
    otsu_df.round(2).to_csv(dirpath.joinpath(f"otsu_{config.METRICS_FILENAME}"))


if __name__ == '__main__':
    main()
