from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector
import torch


def print_accuracy(normal_preds: torch.Tensor, anomaly_preds: list[torch.Tensor]):
    print(f'Correct normal predictions: {sum(normal_preds)}/{len(normal_preds)} -- {sum(normal_preds)/len(normal_preds)*100}')

    correct_anomaly = sum([sum(pred) for pred in anomaly_preds])
    total_anomaly = sum([len(pred) for pred in anomaly_preds])
    print(f'Correct seizure predictions: {correct_anomaly}/{total_anomaly} -- {correct_anomaly/total_anomaly*100}\n')


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    X_normal, X_anomalies = load_data(config.H5_PATH, "chb15")
    _, _, X_test, X_anomalies = split_data(X_normal, X_anomalies)

    # Convert to tensor
    X_test, = convert_to_tensor(X_test)
    X_anomalies = list(convert_to_tensor(*X_anomalies))

    model = AnomalyDetector.load(dirpath)
    normal_preds = model.predict(X_test)
    anomaly_preds = [model.predict(x) for x in X_anomalies]
    print_accuracy(normal_preds, anomaly_preds)


# def main():
#     config.SAVED_MODEL_PATH.mkdir(exist_ok=True, parents=True)
#     X_normal, X_anomalies = load_data(config.H5_PATH, "chb15")
#     X_train, X_val, *_ = split_data(X_normal, X_anomalies)
#     sample_length = X_train.shape[2]
#     n_channels = X_train.shape[1]

#     X_train, X_val, X_test = convert_to_tensor(X_train, X_val, X_test)
#     X_anomalies = list(convert_to_tensor(*X_anomalies))
#     # Normalize data
#     print("Normalizing data...")
#     standardizer = Standardizer()
#     X_train = standardizer.fit_transform(X_train)
#     X_val = standardizer.transform(X_val)
#     X_test = standardizer.transform(X_test)
#     X_anomalies = [standardizer.transform(x) for x in X_anomalies]

#     model = Autoencoder(sample_length, config.N_FILTERS, n_channels, config.KERNEL_SIZE, config.N_SUBWINDOWS)
#     model.train_model(X_train, X_val, dirpath=config.SAVED_MODEL_PATH, n_epochs=100)
#     model.save(config.SAVED_MODEL_PATH)

# def main():
#     X_train = load_data(config.H5_PATH, "chb15")
#     X_train, X_test = split_data(X_train)
#     sample_length = X_train.shape[2]
#     n_channels = X_train.shape[1]

#     # X_train = convert_to_tensor(X_train)
#     # X_test = convert_to_tensor(X_test)
#     # # Normalize data
#     # print("Normalizing data...")
#     # standardizer = Standardizer()
#     # X_train = standardizer.fit_transform(X_train)
#     # X_test = standardizer.transform(X_test)
#     model = Autoencoder(sample_length, config.N_FILTERS, n_channels, config.KERNEL_SIZE, config.N_SUBWINDOWS)
#     # model = LSTMAutoencoder(seq_len=12, n_features=128, encoding_dim=64)

#     print(list(m.shape for m in model.parameters()))


#     # print(sum([m.shape[0] if len(m.shape) == 1 else m.shape[0]*m.shape[1] for m in model.parameters()]))
#     # summary(model, (12, 128))
#     # model.train_model(model, X_train, X_test, n_epochs=100)
if __name__ == '__main__':
    main()
