from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector
import torch
from typing import List


def print_accuracy(normal_preds: torch.Tensor, anomaly_preds: List[torch.Tensor]):
    total_normal = len(normal_preds)
    correct_normal = total_normal - sum(normal_preds)
    print(f'Correct normal predictions: {correct_normal}/{total_normal} -- {correct_normal/total_normal*100}')

    correct_anomaly = sum([sum(pred) for pred in anomaly_preds])
    total_anomaly = sum([len(pred) for pred in anomaly_preds])
    print(f'Correct seizure predictions: {correct_anomaly}/{total_anomaly} -- {correct_anomaly/total_anomaly*100}')


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    X_normal, X_anomalies = load_data(config.H5_PATH, "chb15")
    X_train, X_val, X_test, X_anomalies = split_data(X_normal, X_anomalies, random_state=config.RANDOM_STATE)

    # Convert to tensor
    X_train, X_val = convert_to_tensor(X_train, X_val)
    X_test, = convert_to_tensor(X_test)
    X_anomalies = list(convert_to_tensor(*X_anomalies))

    model = AnomalyDetector.load(dirpath)
    model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE, dirpath=dirpath, learning_rate=config.LEARNING_RATE)

    normal_preds = model.predict(X_test)
    anomaly_preds = [model.predict(x) for x in X_anomalies]
    print_accuracy(normal_preds, anomaly_preds)
    model.save(dirpath)


if __name__ == '__main__':
    main()
