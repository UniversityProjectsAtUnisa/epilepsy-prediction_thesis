import numpy as np
from typing import Tuple
from .evaluation_utils import get_time_left
from statistics import mean


def prediction_accuracy(preds, n_positive_windows) -> float:
    correct = 0
    for sample_pred in preds:
        occurrence_indices = np.flatnonzero(sample_pred == 1)
        if len(occurrence_indices) == 0:
            continue
        first_occurrence_index = int(occurrence_indices[0])
        idx_from_end = len(sample_pred) - 1 - first_occurrence_index
        if idx_from_end < n_positive_windows:
            correct += 1.0
    return float(correct / len(preds))


def specificity(preds, n_positive_windows) -> float:
    tn = 0
    fp = 0
    for sample_pred in preds:
        sample_pred = sample_pred[:-n_positive_windows]
        fp += sum(sample_pred)
        tn += len(sample_pred) - sum(sample_pred)

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


def ppv(preds, n_positive_windows):
    tp = 0
    fp = 0
    for sample_pred in preds:
        tp += sum(sample_pred[-n_positive_windows:])
        fp += sum(sample_pred[:-n_positive_windows])
    return float(tp / (tp + fp))


def bppv(preds, n_positive_windows):
    tp = 0
    fp = 0
    p = 0
    n = 0
    for sample_pred in preds:
        tp += sum(sample_pred[-n_positive_windows:])
        fp += sum(sample_pred[:-n_positive_windows])
        p += sum(sample_pred)
        n += len(sample_pred) - p
    return float((tp/p) / ((tp/p) + (fp/n)))


def imbalanceness(preds, n_positive_windows):
    p = 0
    total = 0
    for sample_pred in preds:
        p += min(n_positive_windows, len(sample_pred))
        total += len(sample_pred)
    return float(p/total)
