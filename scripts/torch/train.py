from data_functions import load_data, split_data, convert_to_tensor
from model.modules.standardizer import FunctionalStandardizer
import torch_config as config
from model.autoencoder import Autoencoder


def main():
    X_train = load_data(config.H5_PATH, "chb15")
    X_train, X_test = split_data(X_train)
    sample_length = X_train.shape[2]
    n_channels = X_train.shape[1]

    X_train = convert_to_tensor(X_train)
    X_test = convert_to_tensor(X_test)
    # Normalize data
    print("Normalizing data...")
    standardizer = FunctionalStandardizer()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)
    model = Autoencoder(sample_length, config.N_FILTERS, n_channels, config.KERNEL_SIZE, config.N_SUBWINDOWS)
    model.train_model(model, X_train, X_test, n_epochs=100)


if __name__ == '__main__':
    main()
