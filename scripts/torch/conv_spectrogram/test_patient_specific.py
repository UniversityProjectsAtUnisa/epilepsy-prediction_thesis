from ..data_functions import load_data, convert_to_tensor, load_patient_names, patient_specific_kfolds_conv
from ..evaluation import quality_metrics as qm
from ..evaluation import plot_functions as pf
from .. import torch_config as config
from .model.cnn_anomaly_detector import CNNAnomalyDetector
from ..utils.gpu_utils import device_context
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def average_performances(df, rowname, window_size_seconds, window_overlap_seconds):
    mean_series = df.mean(axis=0)
    mean_series["IFP (s)"] = qm.inter_fp_seconds(mean_series["spec%"]/100, window_size_seconds, window_overlap_seconds)
    return pd.DataFrame(mean_series, columns=[rowname]).T


def evaluate(positive_preds, negative_preds, n_train_windows,
             n_normal_test_windows, window_size_seconds, window_overlap_seconds):
    metrics = {}
    metrics["train (s)"], _, metrics["test_normal (s)"] = qm.normal_files_seconds(
        n_train_windows, 0, n_normal_test_windows, window_size_seconds, window_overlap_seconds)
    metrics['n_seizures'], metrics["ASD (s)"] = qm.seizure_info(positive_preds, window_size_seconds, window_overlap_seconds)
    # metrics['undetected%'] = 100*qm.undetected_predictions(positive_preds)
    metrics['pred%'] = 100*qm.prediction_accuracy(positive_preds)
    metrics['spec%'] = 100*qm.specificity(negative_preds)
    metrics["IFP (s)"] = qm.inter_fp_seconds(qm.specificity(negative_preds), window_size_seconds, window_overlap_seconds)
    metrics['APT (s)'], metrics['mPT (s)'], metrics['MPT (s)'] = qm.prediction_seconds_before_seizure(
        positive_preds, window_size_seconds, window_overlap_seconds)
    metrics['PPV (%)'] = 100*qm.ppv(positive_preds, negative_preds)
    metrics['BPPV (%)'] = 100*qm.bppv(positive_preds, negative_preds)
    metrics['Imb. (%)'] = 100*qm.imbalanceness(positive_preds, negative_preds)
    return metrics


def main():
    dirpath = config.SAVED_MODEL_DIRPATH
    dirpath.mkdir(exist_ok=True, parents=True)

    patient_names = sorted(load_patient_names(config.H5_FILEPATH))

    metrics_df = pd.DataFrame(columns=config.METRIC_NAMES)

    for patient_name in patient_names:
        print(f"Testing for patient {patient_name}")
        patient_dirpath = dirpath/patient_name
        X_normal, X_anomalies = load_data(config.H5_FILEPATH, patient_name, n_subwindows=config.WINDOW_SIZE_SECONDS,
                                          overlap=config.WINDOW_OVERLAP_SECONDS, preprocess=False)
        if not X_normal:
            raise ValueError("No training data found")

        if not X_anomalies:
            raise ValueError("No test data found")

        with device_context:
            X_normal = convert_to_tensor(*X_normal)
            X_anomalies = convert_to_tensor(*X_anomalies)

        foldmetrics_df = pd.DataFrame(columns=config.METRIC_NAMES)
        fold_positive_preds, fold_negative_preds = [], []

        for input_fold_dirpath in (d for d in patient_dirpath.iterdir() if d.is_dir()):
            print(f"Testing for patient {patient_name} - input fold {input_fold_dirpath.name}")
            with device_context:
                model = CNNAnomalyDetector.load(input_fold_dirpath, svm_dirpath=None)
                embeddings = tuple(model.calculate_embeddings(sequence).cpu().numpy() for sequence in X_normal)
                anom_embeddings = tuple(model.calculate_embeddings(sequence).cpu().numpy() for sequence in X_anomalies)

            for i, (X_train, X_normal_test) in patient_specific_kfolds_conv(embeddings):

                inner_fold_dirpath = input_fold_dirpath/f"i_{i}"
                model.load_svm(inner_fold_dirpath)
                negative_preds = model.svm.predict(X_normal_test) < 0
                positive_preds = tuple(model.svm.predict(x) < 0 for x in anom_embeddings)

                fold_negative_preds.append(negative_preds)
                fold_positive_preds.append(positive_preds)
                # save metrics for folds
                metrics_dict = evaluate(
                    positive_preds, negative_preds, X_train.shape[0],
                    X_normal_test.shape[0],
                    config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
                new_metrics_df = pd.DataFrame([metrics_dict], index=[f"{input_fold_dirpath.name}_{inner_fold_dirpath.name}"])
                foldmetrics_df = pd.concat([foldmetrics_df, new_metrics_df])

        average_row = average_performances(foldmetrics_df, "average", config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
        foldmetrics_df = pd.concat([foldmetrics_df, average_row])
        average_row.index = [patient_name]  # type: ignore
        metrics_df = pd.concat([metrics_df, average_row])
        foldmetrics_df.round(1).to_csv(patient_dirpath/f"{config.METRICS_FILENAME}")

        plot = pf.plot_cumulative_negative_preds(fold_negative_preds, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
        plot.savefig(str(patient_dirpath/f"negative_{config.CUMULATIVE_PREDICTIONS_FILENAME}"))
        plt.close(plot)
        plot = pf.plot_cumulative_positive_preds(fold_positive_preds, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
        plot.savefig(str(patient_dirpath/f"positive_{config.CUMULATIVE_PREDICTIONS_FILENAME}"))
        plt.close(plot)
        print()

    average_row = average_performances(metrics_df, "average", config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
    metrics_df = pd.concat([metrics_df, average_row])
    metrics_df.round(1).to_csv(dirpath/f"{config.METRICS_FILENAME}")


if __name__ == '__main__':
    main()
