from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from model.autoencoder import Autoencoder
from model.standardizer import Standardizer
from model.threshold import Threshold


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    X_normal, X_anomalies = load_data(config.H5_PATH, "chb15")
    X_train, X_val, *_ = split_data(X_normal, X_anomalies, random_state=config.RANDOM_STATE)

    # Convert to tensor
    X_train, X_val = convert_to_tensor(X_train, X_val)
    # X_anomalies = list(convert_to_tensor(*X_anomalies))

    model_path = dirpath.joinpath(AnomalyDetector.model_dirname)
    standardizer_path = dirpath.joinpath(AnomalyDetector.standardizer_dirname)

    with device_context:
        sample_length = X_train.shape[2]
        n_channels = X_train.shape[1]
        model = Autoencoder.load_from_checkpoint(model_path, sample_length, config.N_FILTERS, n_channels, config.KERNEL_SIZE, config.N_SUBWINDOWS)
        standardizer = Standardizer.load(standardizer_path)
        X_train = standardizer.transform(X_train)

        losses_train = model.calculate_losses(X_train)
        threshold = Threshold()
        threshold.fit(losses_train)

        model = AnomalyDetector(model, standardizer, threshold)

    model.save(dirpath)


if __name__ == '__main__':
    main()
