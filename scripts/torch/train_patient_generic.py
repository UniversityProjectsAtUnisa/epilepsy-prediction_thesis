import torch_config as config
from data_functions import convert_to_tensor, load_numpy_dataset, load_patient_names, patient_generic_kfolds
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from evaluation import plot_functions as pf
from utils.train_utils import ConditionalParallelTrainer
import torch


def train(patient_name, other_patients, dirpath):
    print(f"Training for patient {patient_name}")
    patient_dirpath = dirpath.joinpath(patient_name)
    if (patient_dirpath/"complete").exists():
        print(f"Patient {patient_name} already trained")
        return

    X_normals = []
    for p in other_patients:
        X_normal, _ = load_numpy_dataset(config.H5_FILEPATH, p, load_test=False, n_subwindows=config.N_SUBWINDOWS, preprocess=not config.USE_CONVOLUTION)
        if X_normal:
            X_normals.append(X_normal)

    if not X_normals:
        raise ValueError("No training data found")

    for i, (X_train, X_val) in patient_generic_kfolds(X_normals):
        print(f"Training for patient {patient_name} - fold {i}")
        fold_dirpath = patient_dirpath/f"fold_{i}"

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
