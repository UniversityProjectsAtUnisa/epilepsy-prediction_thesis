import torch_config as config
from data_functions import convert_to_tensor, load_numpy_dataset, load_patient_names, interpatient_nested_kfolds
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from evaluation import plot_functions as pf
from multiprocessing import Pool
import torch
from utils.train_utils import ConditionalParallelTrainer
from utils.types import FinetuningMode


def prepare_for_finetuning(model, mode: FinetuningMode):
    if mode == FinetuningMode.FULL:
        return model

    if mode == FinetuningMode.FULL_RESETLASTLAYER:
        model.model.lstm_autoencoder.decoder.output_layer.reset_parameters()
        return model

    if mode == FinetuningMode.DECODER:
        for p in model.model.parameters():
            p.requires_grad = False
        for p in model.model.lstm_autoencoder.decoder.rnn1.parameters():
            p.requires_grad = True
        for p in model.model.lstm_autoencoder.decoder.rnn2.parameters():
            p.requires_grad = True
        for p in model.model.lstm_autoencoder.decoder.output_layer.parameters():
            p.requires_grad = True
        return model


def train(patient_name, dirpath, patientgeneric_dirpath, finetuning_mode: FinetuningMode):
    print(f"Training for patient {patient_name}")
    input_patient_dirpath = patientgeneric_dirpath.joinpath(patient_name)
    if not (input_patient_dirpath/"complete").exists():
        print(f"Patient generic model for patient {patient_name} has not been trained yet")
        return
    output_patient_dirpath = dirpath.joinpath(patient_name)
    if (output_patient_dirpath/"complete").exists():
        print(f"Interpatient model for patient {patient_name} has already been trained")
        return

    X_normal, _ = load_numpy_dataset(config.H5_FILEPATH, patient_name, load_test=True,
                                     n_subwindows=config.N_SUBWINDOWS, preprocess=not config.USE_CONVOLUTION)

    if not X_normal:
        raise ValueError("No training data found")

    for input_fold_dirpath, ei, ii, (X_train, X_val, X_test) in interpatient_nested_kfolds(X_normal, input_patient_dirpath):
        print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - external fold {ei} - internal fold {ii}")
        output_fold_dirpath = output_patient_dirpath/f"{input_fold_dirpath.name}_ei_{ei}_ii_{ii}"

        with device_context:
            X_train, X_val, X_test = convert_to_tensor(X_train, X_val, X_test)
            model = AnomalyDetector.load(input_fold_dirpath)
            model = prepare_for_finetuning(model, finetuning_mode)
            
            history = model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE,
                                  dirpath=output_fold_dirpath, learning_rate=config.LEARNING_RATE)
        model.save(output_fold_dirpath)
        pf.plot_history(history, output_fold_dirpath/config.LOSS_PLOT_FILENAME)
    open(output_patient_dirpath/"complete", "w").close()
    print()


def main():
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    patientgeneric_dirpath = config.PATIENT_GENERIC_OUTPUT_PATH
    finetuning_mode = config.FINETUNING_MODE

    patient_names = load_patient_names(config.H5_FILEPATH)

    trainer = ConditionalParallelTrainer(train_function=train, parallel_training=config.PARALLEL_TRAINING, n_workers=config.PARALLEL_WORKERS)
    args = [(patient_name, dirpath, patientgeneric_dirpath, finetuning_mode)
            for patient_name in patient_names
            if patient_name not in config.SKIP_PATIENTS]

    trainer(*args)


if __name__ == '__main__':
    main()
