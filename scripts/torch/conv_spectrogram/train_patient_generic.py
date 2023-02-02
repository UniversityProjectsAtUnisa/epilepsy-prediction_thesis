from .. import torch_config as config
from ..data_functions import convert_to_tensor, load_data_for_traditional_training, load_patient_names, patient_generic_kfolds_h5py
from .model.cnn_classifier import CNNClassifier
from ..utils.gpu_utils import device_context
from ..evaluation import plot_functions as pf
from ..utils.train_utils import ConditionalParallelTrainer

import h5py
import pathlib


def train(patient_name, other_patients, dirpath):
    print(f"Training for patient {patient_name}")
    patient_dirpath = dirpath/patient_name
    patient_dirpath.mkdir(exist_ok=True, parents=True)
    if (patient_dirpath/"complete").exists():
        print(f"Patient {patient_name} already trained")
        return

    cache_filepath = pathlib.Path(f"{patient_dirpath}/cache.h5")

    if not cache_filepath.exists():
        c = 0
        with h5py.File(cache_filepath, "a") as f:
            for p in other_patients:
                X, y = load_data_for_traditional_training(config.H5_FILEPATH, p, preictal_seconds=config.PREICTAL_SECONDS,
                                                          n_subwindows=config.WINDOW_SIZE_SECONDS, overlap=config.WINDOW_OVERLAP_SECONDS)
                f.create_dataset(f"X/{c}", data=X)
                f.create_dataset(f"y/{c}", data=y)
                c += 1

    with h5py.File(cache_filepath) as f:
        for i, (X_train, y_train, X_val, y_val) in patient_generic_kfolds_h5py(f["X"], f["y"]):
            print(f"Training for patient {patient_name} - fold {i}")
            fold_dirpath = patient_dirpath/f"fold_{i}"

            X_train, y_train, X_val, y_val = convert_to_tensor(X_train, y_train, X_val, y_val)
            with device_context:
                model = CNNClassifier()
                history = model.train_model(X_train, y_train, X_val, y_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE,
                                            dirpath=fold_dirpath, learning_rate=config.LEARNING_RATE, patience=config.PATIENCE)

            model.save(fold_dirpath)
            pf.plot_history(history, fold_dirpath/config.LOSS_PLOT_FILENAME)
    open(patient_dirpath/"complete", "w").close()
    print()


def main():
    dirpath = config.SAVED_MODEL_DIRPATH
    dirpath.mkdir(exist_ok=True, parents=True)

    patient_names = load_patient_names(config.H5_FILEPATH)

    trainer = ConditionalParallelTrainer(train_function=train, parallel_training=config.PARALLEL_TRAINING, n_workers=config.PARALLEL_WORKERS)

    args = []
    for patient_name in patient_names:
        if patient_name in config.SKIP_PATIENTS:
            continue
        args.append((patient_name, [p for p in patient_names if p != patient_name], dirpath))
    trainer(*args)


if __name__ == '__main__':
    main()
