import torch_config as config
from data_functions import convert_to_tensor, load_data, split_data, load_patient_names
from model.anomaly_detector import AnomalyDetector
from utils.gpu_utils import device_context
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import torch
import pandas as pd

x_name_normal = 'normal_1_0_data_x.npy'
y_name_normal = 'normal_1_0_data_y.npy'
x_name_seizure = 'seizure_1_0_data_x.npy'
y_name_seizure = 'seizure_1_0_data_y.npy'


def paper_convert_to_tensor(data):
    sequences = data.tolist()
    #dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    dataset = torch.tensor(data).float()
    n_seq, n_features = dataset.shape
    return dataset, n_features


def paper_load_data(directory, x_name, y_name, conf_channel=3, pandas=False, seql=0):

    print("Loading x_data... ", end=" ")

    # x_data = np.load(directory/x_name, allow_pickle=True)  # datos

    # if conf_channel == 1:
    #     data_x = x_data[:, 5]-x_data[:, 9]  # resta del canal 6 y 10, ofrecen más información sobre ataques epilepticos
    # elif conf_channel == 2:
    #     data_x = x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2])  # concatenacion 21 canales
    # elif conf_channel == 3:
    #     data_x = x_data.mean(axis=1)  # media aritmetica canales

    # else:
    #     print('ERROR LOAD TYPE!!')

    # print(f'DONE  Shape: {x_data.shape}')

    print("Loading y_data... ", end=" ")
    y_data = np.load(directory/y_name, allow_pickle=True)  # metadatos
    data_y = y_data
    #data_y = y_data [:,7].astype('int')
    print(f'DONE  Shape: {y_data.shape}')

    print("Tranform data... ", end=" ")

    if pandas:
        print("Loading pandas data_x... ")
        # data_x = pd.DataFrame(data_x)
        print("Loading pandas data_y... ")
        data_y = pd.DataFrame(data_y, columns=['type', 'name', 'filename', 'pre1', 'pre2', 'id_eeg_actual', 'id_eeg_all', 'label'])
        data_y.label = data_y.label.astype('int')
        return None, data_y

    #rus = RandomUnderSampler(sampling_strategy=0.5)
    #data_x_r, data_y_r= rus.fit_resample(data_x, data_y)

    # tranformamos [NSamples x 128] a [NSamples x LSeq x 128]
    if seql != 0:
        nSamp = int(data_y.shape[0]/seql)
        # data_x = data_x[:nSamp*seql].reshape(-1, 128*seql)
        data_y = data_y[:nSamp*seql, -1].reshape(-1, seql)
    data_x = None
    # print(f'DONE  Shape: {data_x.shape} \n')

    return data_x, data_y


def train(patient_name):
    dirpath = config.SAVED_MODEL_PATH
    print(f"Training for patient {patient_name}")
    patient_dirpath = dirpath.joinpath(patient_name)
    if (patient_dirpath/"complete").exists():
        print(f"Patient {patient_name} already trained")
        return
    directory = directory = Path('/media/marco741/Archive/Datasets/EEG data')
    _, data_y_normal = paper_load_data(directory, x_name_normal, y_name_normal, conf_channel=3, pandas=True, seql=2)
    _, data_y_seizure = paper_load_data(directory, x_name_seizure, y_name_seizure,
                                        conf_channel=3, pandas=True, seql=2)
    # data_x_seizure = data_x_seizure[(np.unique(np.where(data_y_seizure[:, 0] == 1)[0]))]
    train_normal, test_normal = train_test_split(data_x_normal, test_size=0.20, random_state=42)
    X_test_ictal, _ = paper_convert_to_tensor(data_x_seizure)
    train_normal, _ = paper_convert_to_tensor(train_normal)
    test_normal, _ = paper_convert_to_tensor(test_normal)
    # X_normal, _, X_test_ictal = load_data(config.H5_FILEPATH, patient_name, load_test=True, preprocess=not config.USE_CONVOLUTION)

    # X_normal = np.concatenate([*X_normal, X_test_normal])
    # X_train, X_val = split_data(X_normal, random_state=config.RANDOM_STATE)
    # X_train, X_val = train_test_split(X_normal, train_size=0.8, random_state=config.RANDOM_STATE)

# Convert to tensor

    with device_context:
        # X_train, X_val, X_test_ictal = convert_to_tensor(train_normal, test_normal, X_test_ictal)  # type: ignore
        X_test_ictal = X_test_ictal.to(device_context.device)
        train_normal = train_normal.to(device_context.device)
        test_normal = test_normal.to(device_context.device)
        model = AnomalyDetector(use_convolution=config.USE_CONVOLUTION)
        model.train(train_normal, test_normal, X_test_ictal, n_epochs=config.N_EPOCHS, batch_size=config.BATCH_SIZE,
                    dirpath=patient_dirpath, learning_rate=config.LEARNING_RATE)

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
