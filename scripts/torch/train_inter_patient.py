import torch_config as config
from data_functions import convert_to_tensor, load_numpy_dataset, load_patient_names, interpatient_nested_kfolds
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from evaluation import plot_functions as pf
import numpy as np
from collections import defaultdict
from statistics import mean
from multiprocessing import Pool
import torch


def prepare_for_finetuning(model):
    # for p in model.model.parameters():
    #     p.requires_grad = True
    # for p in model.model.lstm_autoencoder.encoder.rnn1.parameters():
    #     p.requires_grad = True
    # for p in model.model.lstm_autoencoder.encoder.rnn2.parameters():
    #     p.requires_grad = True
    # for p in model.model.lstm_autoencoder.decoder.rnn1.parameters():
    #     p.requires_grad = True
    # for p in model.model.lstm_autoencoder.decoder.rnn2.parameters():
    #     p.requires_grad = True
    # for p in model.model.lstm_autoencoder.decoder.output_layer.parameters():
    #     p.requires_grad = True
    model.model.lstm_autoencoder.decoder.output_layer.reset_parameters()
    return model


def train(patient_name, dirpath, patientgeneric_dirpath):
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

    # models = defaultdict(list)
    # thresholds = {}

    # max_val_losses, max_test_losses = [], []

    # for input_fold_dirpath, ei, ii, (X_train, X_val, X_test) in interpatient_nested_kfolds(X_normal, input_patient_dirpath):
    #     print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - external fold {ei} - internal fold {ii}")
    #     output_fold_dirpath = output_patient_dirpath/f"{input_fold_dirpath.name}_ei_{ei}_ii_{ii}"
    #     k = f"{input_fold_dirpath.name}_ei_{ei}"
    #     with device_context:
    #         X_val, X_test = convert_to_tensor(X_val, X_test)
    #         m = AnomalyDetector.load(output_fold_dirpath)
    #         val_losses = m.model.calculate_losses(X_val)
    #         test_losses = m.model.calculate_losses(X_test)
    #         m.threshold.threshold = float(max(val_losses))
    #     max_val_losses.append(m.threshold.threshold)
    #     max_test_losses.append(float(max(test_losses)))
    #     models[0].append(m)

    # for k, list_of_models in models.items():
    #     new_th = max(m.threshold.threshold for m in list_of_models)
    #     thresholds[k] = new_th

    # for input_fold_dirpath, ei, ii, (X_train, X_val, X_test) in interpatient_nested_kfolds(X_normal, input_patient_dirpath):
    #     print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - external fold {ei} - internal fold {ii}")
    #     output_fold_dirpath = output_patient_dirpath/f"{input_fold_dirpath.name}_ei_{ei}_ii_{ii}"

    #     k = f"{input_fold_dirpath.name}_ei_{ei}"
    #     with device_context:
    #         m = AnomalyDetector.load(output_fold_dirpath)
    #         X_val, X_test = convert_to_tensor(X_val, X_test)
    #         val_losses = m.model.calculate_losses(X_val)
    #         # m.threshold.fit(val_losses)
    #         # m.threshold.threshold = float(max(val_losses))
    #         # old_th = m.threshold.threshold
    #         test_losses = m.model.calculate_losses(X_test)
    #         # m.threshold.fit(test_losses)
    #         m.threshold.threshold = thresholds[0]
    #     m.save(output_fold_dirpath)
    #     # models[k].append(m)

    # thresholds = {}
    # for k, list_of_models in models.items():
    #     new_th = max(m.threshold.threshold for m in list_of_models)
    #     thresholds[k] = new_th

    # for input_fold_dirpath, ei, ii, (X_train, X_val, X_test) in interpatient_nested_kfolds(X_normal, input_patient_dirpath):
    #     print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - external fold {ei} - internal fold {ii}")
    #     output_fold_dirpath = output_patient_dirpath/f"{input_fold_dirpath.name}_ei_{ei}_ii_{ii}"

    #     m = AnomalyDetector.load(output_fold_dirpath)
    #     k = f"{input_fold_dirpath.name}_ei_{ei}"
    #     m.threshold.threshold = thresholds[k]
    #     m.save(output_fold_dirpath)

    # specs = []
    # for input_fold_dirpath, ei, ii, (X_train, X_val, X_test) in interpatient_nested_kfolds(X_normal, input_patient_dirpath):
    #     print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - external fold {ei} - internal fold {ii}")
    #     output_fold_dirpath = output_patient_dirpath/f"{input_fold_dirpath.name}_ei_{ei}_ii_{ii}"
    #     with device_context:
    #         X_train, X_val, X_test = convert_to_tensor(X_train, X_val, X_test)
    #         model = AnomalyDetector.load(output_fold_dirpath)
    #         norm_pred = model.predict(X_test)
    #         norm_pred2 = model.predict(X_val)
    #         anom_pred = model.predict(X_anomalies)
    #         specs.append(float(1 - sum(norm_pred)/len(norm_pred)))
    #     print(f"spec {1 - sum(norm_pred)/len(norm_pred)} - valspec {1 - sum(norm_pred2)/len(norm_pred2)} - anom acc {sum(anom_pred)/len(anom_pred)}")
    # print(f"mean spec {np.mean(specs)}")

    for input_fold_dirpath, ei, ii, (X_train, X_val, X_test) in interpatient_nested_kfolds(X_normal, input_patient_dirpath):
        print(f"Training for patient {patient_name} - input fold {input_fold_dirpath.name} - external fold {ei} - internal fold {ii}")
        output_fold_dirpath = output_patient_dirpath/f"{input_fold_dirpath.name}_ei_{ei}_ii_{ii}"

        with device_context:
            X_train, X_val, X_test = convert_to_tensor(X_train, X_val, X_test)
            model = AnomalyDetector.load(input_fold_dirpath)

            model = prepare_for_finetuning(model)
            history = model.train(X_train, X_val, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE,
                                  dirpath=output_fold_dirpath, learning_rate=config.LEARNING_RATE)
        model.save(output_fold_dirpath)
        pf.plot_history(history, output_fold_dirpath/config.LOSS_PLOT_FILENAME)
    open(output_patient_dirpath/"complete", "w").close()
    print()


def main():
    torch.multiprocessing.set_start_method('spawn')
    dirpath = config.SAVED_MODEL_PATH
    dirpath.mkdir(exist_ok=True, parents=True)
    patientgeneric_dirpath = config.PATIENT_GENERIC_OUTPUT_PATH

    if config.PATIENT_ID:
        patient_names = [config.PATIENT_ID]
    else:
        patient_names = load_patient_names(config.H5_FILEPATH)

    with Pool(3) as p:
        args = []
        for patient_name in patient_names:
            if patient_name in config.SKIP_PATIENTS:
                continue
            args.append((patient_name, dirpath, patientgeneric_dirpath))
        p.starmap(train, args)

    # for patient_name in patient_names:
    #     if patient_name in config.SKIP_PATIENTS:
    #         continue
    #     train(patient_name, dirpath, patientgeneric_dirpath)


if __name__ == '__main__':
    main()
