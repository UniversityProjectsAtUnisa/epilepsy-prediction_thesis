from .. import torch_config as config
from ..data_functions import convert_to_tensor, load_data, load_patient_names, patient_specific_kfolds_conv
from .model.cnn_anomaly_detector import CNNAnomalyDetector
from ..utils.gpu_utils import device_context
from ..evaluation import plot_functions as pf
from ..utils.train_utils import ConditionalParallelTrainer


def train(patient_name, dirpath):
    print(f"Training for patient {patient_name}")
    if not (dirpath/"complete").exists():
        print(f"Patient generic model for patient {patient_name} has not been trained yet")
        return
    if (dirpath/"svmcomplete").exists():
        print(f"Interpatient models for patient {patient_name} have already been trained")
        return

    X_normal, _ = load_data(config.H5_FILEPATH, patient_name, load_test=False, n_subwindows=config.WINDOW_SIZE_SECONDS,
                            overlap=config.WINDOW_OVERLAP_SECONDS, preprocess=False)

    if not X_normal:
        raise ValueError("No training data found")

    with device_context:
        X_normal = convert_to_tensor(*X_normal)

    for input_fold_dirpath in (d for d in dirpath.iterdir() if d.is_dir()):
        print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name}")
        with device_context:
            model = CNNAnomalyDetector.load(input_fold_dirpath, svm_dirpath=None)
            embeddings = tuple(model.calculate_embeddings(sequence).cpu().numpy() for sequence in X_normal)

        for i, (X_train, _) in patient_specific_kfolds_conv(embeddings):
            print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - output fold {i}")
            output_fold_dirpath = input_fold_dirpath/f"i_{i}"
            model.train(X_train)
            model.save_svm(output_fold_dirpath)
    open(dirpath/"svmcomplete", "w").close()
    print()


def main():
    dirpath = config.SAVED_MODEL_DIRPATH
    dirpath.mkdir(exist_ok=True, parents=True)

    patient_names = load_patient_names(config.H5_FILEPATH)

    trainer = ConditionalParallelTrainer(train_function=train, parallel_training=config.PARALLEL_TRAINING, n_workers=config.PARALLEL_WORKERS)
    args = [(patient_name, dirpath/patient_name)
            for patient_name in patient_names
            if patient_name not in config.SKIP_PATIENTS]

    trainer(*args)


if __name__ == '__main__':
    main()
