from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    X_normal, X_anomalies = load_data(config.H5_PATH, "chb15")
    X_train, X_val, *_ = split_data(X_normal, X_anomalies, random_state=config.RANDOM_STATE)

    # Convert to tensor
    X_train, X_val = convert_to_tensor(X_train, X_val)
    # X_anomalies = list(convert_to_tensor(*X_anomalies))

    with device_context:
        model = AnomalyDetector()
        model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE, dirpath=dirpath, learning_rate=config.LEARNING_RATE)

    model.save(dirpath)


if __name__ == '__main__':
    main()
