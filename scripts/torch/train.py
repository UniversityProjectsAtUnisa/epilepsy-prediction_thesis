from data_functions import load_data, split_data, convert_to_tensor
import torch_config as config
from model.anomaly_detector import AnomalyDetector


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    # Load data
    X_normal, X_anomalies = load_data(config.H5_PATH, "chb15")
    X_train, X_val, *_ = split_data(X_normal, X_anomalies, random_state=config.RANDOM_STATE)

    # Convert to tensor
    X_train, X_val = convert_to_tensor(X_train, X_val)
    # X_anomalies = list(convert_to_tensor(*X_anomalies))

    model = AnomalyDetector()
    model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE, dirpath=dirpath, learning_rate=config.LEARNING_RATE)

    model.save(dirpath)


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
