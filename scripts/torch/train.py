import torch_config as config
from data_functions import convert_to_tensor, load_data, split_data, load_patient_names
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from multiprocessing import Pool


def train(patient_name):
    dirpath = config.SAVED_MODEL_PATH
    print(f"Training for patient {patient_name}")
    patient_dirpath = dirpath.joinpath(patient_name)
    if (patient_dirpath/"complete").exists():
        print(f"Patient {patient_name} already trained")
        return
    X_normal, _, _ = load_data(config.H5_FILEPATH, patient_name, load_test=False, preprocess=not config.USE_CONVOLUTION)
    if not X_normal:
        raise ValueError("No training data found")
    X_train, X_val = split_data(X_normal, random_state=config.RANDOM_STATE)

    # Convert to tensor

    with device_context:
        X_train, X_val = convert_to_tensor(X_train, X_val)
        model = AnomalyDetector(use_convolution=config.USE_CONVOLUTION)
        model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE, dirpath=patient_dirpath, learning_rate=config.LEARNING_RATE)

    model.save(patient_dirpath)
    open(patient_dirpath/"complete", "w").close()
    print()


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = load_patient_names(config.H5_FILEPATH)

    # with Pool(3) as p:
    #     p.map(train, patient_names)
    for patient_name in patient_names:
        train(patient_name)


if __name__ == '__main__':
    main()
