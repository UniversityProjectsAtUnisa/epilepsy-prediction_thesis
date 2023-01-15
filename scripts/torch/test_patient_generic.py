from data_functions import load_numpy_dataset, convert_to_tensor, load_patient_names, patient_generic_kfolds
from evaluation import quality_metrics as qm
from evaluation import plot_functions as pf
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd


def average_performances(df, rowname, window_size_seconds, window_overlap_seconds):
    mean_series = df.mean(axis=0)
    mean_series["IFP (s)"] = qm.inter_fp_seconds(mean_series["spec%"]/100, window_size_seconds, window_overlap_seconds)
    return pd.DataFrame(mean_series, columns=[rowname]).T


def evaluate(positive_preds, negative_preds, n_train_windows, n_val_windows,
             n_normal_test_windows, window_size_seconds, window_overlap_seconds):
    metrics = {}
    metrics["train (s)"], metrics["val (s)"], metrics["test_normal (s)"] = qm.normal_files_seconds(
        n_train_windows, n_val_windows, n_normal_test_windows, window_size_seconds, window_overlap_seconds)
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

    percentiles = np.linspace(99.8, 100, 3, endpoint=True)
    metrics_df = {perc: pd.DataFrame(columns=config.METRIC_NAMES) for perc in percentiles}

    for patient_name in patient_names:
        print(f"Testing for patient {patient_name}")
        patient_dirpath = dirpath.joinpath(patient_name)
        other_patients = [p for p in patient_names if p != patient_name]

        X_normals = []
        for p in other_patients:
            X_normal, _ = load_numpy_dataset(config.H5_FILEPATH, p, load_test=False, n_subwindows=config.N_SUBWINDOWS)
            if X_normal:
                X_normals.append(X_normal)

        if not X_normals:
            raise ValueError("No validation data found")

        X_normal_test, X_anomalies = load_numpy_dataset(config.H5_FILEPATH, patient_name, n_subwindows=config.N_SUBWINDOWS)
        if not X_normal_test:
            raise ValueError("No negative test data found")

        if not X_anomalies:
            raise ValueError("No positive test data found")

        X_normal_test = np.concatenate(X_normal_test)
        X_normal_test, = convert_to_tensor(X_normal_test)
        X_anomalies = convert_to_tensor(*X_anomalies)
        for perc in percentiles:
            foldmetrics_df = pd.DataFrame(columns=config.METRIC_NAMES)
            fold_positive_preds, fold_negative_preds = [], []
            for i, (X_train, X_val) in patient_generic_kfolds(X_normals):
                fold_name = f"fold_{i}"
                fold_dirpath = patient_dirpath/fold_name

                X_val, = convert_to_tensor(X_val,)

                with device_context:
                    model = AnomalyDetector.load(fold_dirpath)
                    losses_val = model.model.calculate_losses(X_val)  # type: ignore

                    model.threshold.threshold = np.percentile(losses_val, perc)  # type: ignore
                    negative_preds = model.predict(X_normal_test)
                    positive_preds = tuple(model.predict(x) for x in X_anomalies)

                fold_negative_preds.append(negative_preds)
                fold_positive_preds.append(positive_preds)
                # save metrics for folds
                metrics_dict = evaluate(
                    positive_preds, negative_preds, X_train.shape[0],
                    X_val.shape[0],
                    X_normal_test.shape[0],
                    config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
                new_metrics_df = pd.DataFrame([metrics_dict], index=[fold_name])
                foldmetrics_df = pd.concat([foldmetrics_df, new_metrics_df])

            average_row = average_performances(foldmetrics_df, "average", config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
            foldmetrics_df = pd.concat([foldmetrics_df, average_row])
            average_row.index = [patient_name]  # type: ignore
            metrics_df[perc] = pd.concat([metrics_df[perc], average_row])
            foldmetrics_df.round(1).to_csv(patient_dirpath/f"{perc}_{config.METRICS_FILENAME}")

            plot = pf.plot_cumulative_positive_preds(fold_positive_preds, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
            plot.savefig(str(patient_dirpath/f"{perc}_positive_{config.CUMULATIVE_PREDICTIONS_FILENAME}"))
            plt.close(plot)
            plot = pf.plot_cumulative_negative_preds(fold_negative_preds, config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
            plot.savefig(str(patient_dirpath/f"{perc}_negative_{config.CUMULATIVE_PREDICTIONS_FILENAME}"))
            plt.close(plot)
        print()

    for k, m in metrics_df.items():
        average_row = average_performances(m, "average", config.WINDOW_SIZE_SECONDS, config.WINDOW_OVERLAP_SECONDS)
        m = pd.concat([m, average_row])
        m.round(1).to_csv(dirpath/f"{k}_{config.METRICS_FILENAME}")


if __name__ == '__main__':
    main()
