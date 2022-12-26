import numpy as np
from typing import Tuple
from .evaluation_utils import get_time_left
from statistics import mean


def normal_files_seconds(n_train_windows, n_val_windows, n_normal_test_windows, window_size_seconds, window_overlap_seconds):
    train_seconds = get_time_left(n_train_windows, window_size_seconds, window_overlap_seconds)
    val_seconds = get_time_left(n_val_windows, window_size_seconds, window_overlap_seconds)
    test_seconds = get_time_left(n_normal_test_windows, window_size_seconds, window_overlap_seconds)
    return float(train_seconds), float(val_seconds), float(test_seconds)


def seizure_info(preds, window_size_seconds, window_overlap_seconds):
    n_seizures = len(preds)
    durations = []
    for sample_pred in preds:
        durations.append(get_time_left(len(sample_pred), window_size_seconds, window_overlap_seconds))
    return n_seizures, float(mean(durations))


def intra_fp_seconds(spec, window_size_seconds, window_overlap_seconds):
    fpr = 1 - spec
    if fpr == 0:
        return float('nan')
    ifp = (window_size_seconds - window_overlap_seconds) / fpr
    return float(ifp)


def prediction_accuracy(preds) -> float:
    correct = sum(any(sample_pred == 1) for sample_pred in preds)
    return float(correct / len(preds))


def specificity(preds) -> float:
    fp = sum(preds)
    tn = len(preds) - sum(preds)

    tn_rate = tn / (tn + fp)
    return float(tn_rate)


def prediction_seconds_before_seizure(preds, window_size_seconds: int, window_overlap_seconds: int) -> Tuple[float, float, float]:
    times = []
    for sample_pred in preds:
        occurrence_indices = np.flatnonzero(sample_pred == 1)
        if len(occurrence_indices) == 0:
            continue
        first_occurrence_index = int(occurrence_indices[0])
        idx_from_end = len(sample_pred) - 1 - first_occurrence_index
        time_left = get_time_left(idx_from_end, window_size_seconds, window_overlap_seconds)
        times.append(time_left)

    if len(times) == 0:
        return float('nan'), float('nan'), float('nan')
    avg_pt = mean(times)
    min_pt = min(times)
    max_pt = max(times)
    return float(avg_pt), float(min_pt), float(max_pt)


def undetected_predictions(preds):
    undetected = 0
    for sample_pred in preds:
        if sum(sample_pred) == 0:
            undetected += 1
    return float(undetected/len(preds))


def ppv(positive_preds, negative_preds):
    tp = 0
    fp = sum(negative_preds)
    for sample_pred in positive_preds:
        tp += sum(sample_pred)
    return float(tp / (tp + fp))


def bppv(positive_preds, negative_preds):
    tp = 0
    p = 0
    n = len(negative_preds)
    fp = sum(negative_preds)
    for sample_pred in positive_preds:
        tp += sum(sample_pred)
        p += len(sample_pred)
    return float((tp/p) / ((tp/p) + (fp/n)))


def imbalanceness(positive_preds, negative_preds):
    p = 0
    n = len(negative_preds)
    for sample_pred in positive_preds:
        p += len(sample_pred)
    return float(p/(p+n))
