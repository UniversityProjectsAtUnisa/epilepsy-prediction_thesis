from data_functions import load_numpy_dataset, convert_to_tensor, load_patient_names, nested_kfolds
from evaluation import quality_metrics as qm
from evaluation import plot_functions as pf
from model.helpers.history import History
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from model.autoencoder import Autoencoder
from utils.gpu_utils import device_context
from skimage.filters.thresholding import threshold_otsu
import matplotlib.pyplot as plt


import numpy as np
import math
import pandas as pd


def get_time_left(windows_left: int):
    return windows_left * (config.WINDOW_SIZE_SECONDS - config.WINDOW_OVERLAP_SECONDS)


def average_performances(df, rowname, window_size_seconds, window_overlap_seconds):
    mean_series = df.mean(axis=0)
    mean_series["IFP (s)"] = qm.intra_fp_seconds(mean_series["spec%"]/100, window_size_seconds, window_overlap_seconds)
    return pd.DataFrame(mean_series, columns=[rowname]).T


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


def evaluate(positive_preds, negative_preds, n_train_windows, n_val_windows,
             n_normal_test_windows, window_size_seconds, window_overlap_seconds):
    metrics = {}
    metrics["train (s)"], metrics["val (s)"], metrics["test_normal (s)"] = qm.normal_files_seconds(
        n_train_windows, n_val_windows, n_normal_test_windows, window_size_seconds, window_overlap_seconds)
    metrics['n_seizures'], metrics["ASD (s)"] = qm.seizure_info(positive_preds, window_size_seconds, window_overlap_seconds)
    # metrics['undetected%'] = 100*qm.undetected_predictions(positive_preds)
    metrics['pred%'] = 100*qm.prediction_accuracy(positive_preds)
    metrics['spec%'] = 100*qm.specificity(negative_preds)
    metrics["IFP (s)"] = qm.intra_fp_seconds(qm.specificity(negative_preds), window_size_seconds, window_overlap_seconds)
    metrics['APT (s)'], metrics['mPT (s)'], metrics['MPT (s)'] = qm.prediction_seconds_before_seizure(
        positive_preds, window_size_seconds, window_overlap_seconds)
    metrics['PPV (%)'] = 100*qm.ppv(positive_preds, negative_preds)
    metrics['BPPV (%)'] = 100*qm.bppv(positive_preds, negative_preds)
    metrics['Imb. (%)'] = 100*qm.imbalanceness(positive_preds, negative_preds)
    return metrics


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = sorted(load_patient_names(config.H5_FILEPATH))

    percentiles = np.linspace(99.8, 100, 3, endpoint=True)
    metrics_df = {perc: pd.DataFrame(columns=config.METRIC_NAMES) for perc in percentiles}

    for patient_name in patient_names:
        print(f"Testing for patient {patient_name}")
        patient_dirpath = dirpath.joinpath(patient_name)
        X_normal, X_anomalies = load_numpy_dataset(config.H5_FILEPATH, patient_name, n_subwindows=config.N_SUBWINDOWS, preprocess=not config.USE_CONVOLUTION)
        if not X_normal:
            raise ValueError("No training data found")

        if not X_anomalies:
            raise ValueError("No test data found")

        X_anomalies = convert_to_tensor(*X_anomalies)
        for perc in percentiles:
            foldmetrics_df = pd.DataFrame(columns=config.METRIC_NAMES)
            fold_preds = []
            for ei, ii, (X_train, X_val, X_normal_test) in nested_kfolds(X_normal):
                fold_name = f"ei_{ei}_ii_{ii}"
                fold_dirpath = patient_dirpath/fold_name

                X_val, X_normal_test = convert_to_tensor(X_val, X_normal_test)

                with device_context:
                    model = AnomalyDetector.load(fold_dirpath)
                    losses_val = model.model.calculate_losses(X_val)  # type: ignore

                    model.threshold.threshold = np.percentile(losses_val, perc)  # type: ignore
                    negative_preds = model.predict(X_normal_test)
                    positive_preds = tuple(model.predict(x) for x in X_anomalies)
                    fold_preds.append(positive_preds)
                    # for i in range(3):
                    #     print_sample_evaluations(preds, consecutive_windows=i+1)
                    # print()

                    metrics_dict = evaluate(
                        positive_preds, negative_preds, X_train.shape[0],
                        X_val.shape[0],
                        X_normal_test.shape[0],
                        config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
                    new_metrics_df = pd.DataFrame([metrics_dict], index=[fold_name])
                    foldmetrics_df = pd.concat([foldmetrics_df, new_metrics_df])

                # save metrics for folds
                average_row = average_performances(foldmetrics_df, "average", config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
                foldmetrics_df = pd.concat([foldmetrics_df, average_row])
                average_row.index = [patient_name]
                metrics_df[perc] = pd.concat([metrics_df[perc], average_row])
                foldmetrics_df.round(1).to_csv(patient_dirpath/f"{perc}_{config.METRICS_FILENAME}")

            plot = pf.plot_cumulative_preds(fold_preds, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
            plot.savefig(str(patient_dirpath/f"{perc}_{config.CUMULATIVE_PREDICTIONS_FILENAME}"))
            plt.close(plot)
        print()

    for k, m in metrics_df.items():
        average_row = average_performances(m, "average", config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
        m = pd.concat([m, average_row])
        m.round(1).to_csv(dirpath/f"{k}_{config.METRICS_FILENAME}")


if __name__ == '__main__':
    main()
