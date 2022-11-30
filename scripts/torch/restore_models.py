from data_functions import load_data, convert_to_tensor, load_patient_names, split_data
from evaluation import quality_metrics as qm
from evaluation import plot_functions as pf
from model.helpers.history import History
from model.threshold import Threshold
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from model.autoencoder import Autoencoder
from utils.gpu_utils import device_context

import numpy as np
import math
import pandas as pd


def main():
    # TODO: change to work with convolution enabled
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = sorted(load_patient_names(config.H5_FILEPATH))

    for patient_name in patient_names:
        print(f"Testing for patient {patient_name}")
        patient_dirpath = dirpath.joinpath(patient_name)
        X_normal, X_test = load_data(config.H5_FILEPATH, patient_name)
        if not X_normal:
            raise ValueError("No training data found")
        X_train, X_val = split_data(X_normal, random_state=config.RANDOM_STATE)

        if X_test is None:
            raise ValueError("No test data found")

        X_val, = convert_to_tensor(X_val)
        X_test = convert_to_tensor(*X_test)

        sample_length = X_train.shape[1]
        model = AnomalyDetector()
        model.threshold = Threshold.load(patient_dirpath/model.threshold_filename)
        model.model = Autoencoder(sample_length, config.N_SUBWINDOWS)
        model.model.load_checkpoint(patient_dirpath/model.model_dirname)
        preds = tuple(model.predict(x) for x in X_test)
        model.save(patient_dirpath)

        print()


if __name__ == '__main__':
    main()
