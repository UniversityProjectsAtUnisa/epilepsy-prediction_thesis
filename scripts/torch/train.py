import torch_config as config
from data_functions import convert_to_tensor, load_numpy_dataset, load_patient_names, nested_kfolds
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from evaluation import plot_functions as pf


def train(patient_name, dirpath):
    print(f"Training for patient {patient_name}")
    patient_dirpath = dirpath.joinpath(patient_name)
    if (patient_dirpath/"complete").exists():
        print(f"Patient {patient_name} already trained")
        return
    X_normal, _ = load_numpy_dataset(config.H5_FILEPATH, patient_name, load_test=False, n_subwindows=config.N_SUBWINDOWS, preprocess=not config.USE_CONVOLUTION)

    if not X_normal:
        raise ValueError("No training data found")

    for ei, ii, (X_train, X_val, _) in nested_kfolds(X_normal):
        print(f"Training for patient {patient_name} - external fold {ei} - internal fold {ii}")
        fold_dirpath = patient_dirpath/f"ei_{ei}_ii_{ii}"

        with device_context:
            X_train, X_val = convert_to_tensor(X_train, X_val)
            model = AnomalyDetector(use_convolution=config.USE_CONVOLUTION)
            history = model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE,
                                  dirpath=fold_dirpath, learning_rate=config.LEARNING_RATE)

        model.save(fold_dirpath)
        pf.plot_history(history, fold_dirpath/config.LOSS_PLOT_FILENAME)
    open(patient_dirpath/"complete", "w").close()
    print()


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = load_patient_names(config.H5_FILEPATH)

    for patient_name in patient_names:
        if patient_name in config.SKIP_PATIENTS:
            continue
        train(patient_name, dirpath)


if __name__ == '__main__':
    main()
