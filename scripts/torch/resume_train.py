import torch_config as config
from data_functions import convert_to_tensor, load_data, split_data
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    X_normal, _ = load_data(config.H5_PATH.joinpath(config.H5_FILENAME), "chb15", load_test=False)
    if not X_normal:
        raise ValueError("No training data found")
    X_train, X_val = split_data(X_normal, random_state=config.RANDOM_STATE)

    # Convert to tensor
    X_train, X_val = convert_to_tensor(X_train, X_val)

    with device_context:
        model = AnomalyDetector.load(dirpath)
        model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE, dirpath=dirpath, learning_rate=config.LEARNING_RATE)

    model.save(dirpath)


if __name__ == '__main__':
    main()
