from torch import nn
from typing import Dict, Any
import torch
import time
import copy
import numpy as np
import pickle
from modules.conv_encoder import ConvEncoder
from modules.conv_decoder import ConvDecoder
from modules.lstm_autoencoder import LSTMAutoencoder

N_FILTERS = 3
N_CHANNELS = 18
SAMPLE_LENGTH = 1536
KERNEL_SIZE = 3
LEN_SUBWINDOWS = 128


def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'(Using device: {device})', end=" ")
    return device


class Autoencoder(nn.Module):

    def __init__(self, sample_length=SAMPLE_LENGTH, n_filters=N_FILTERS, n_channels=N_CHANNELS, kernel_size=KERNEL_SIZE, len_subwindows=LEN_SUBWINDOWS):
        super(Autoencoder, self).__init__()
        self.n_channels = n_channels

        self.device = check_device()

        encoding_dim = sample_length//2
        n_subwindows = sample_length//len_subwindows

        self.conv_encoder = ConvEncoder(sample_length=sample_length, n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size)
        self.conv_decoder = ConvDecoder(sample_length=sample_length, n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size)
        self.lstm_autoencoders = [LSTMAutoencoder(seq_len=n_subwindows, n_features=len_subwindows, encoding_dim=encoding_dim) for _ in range(n_filters)]

    def forward(self, x):
        x = self.conv_encoder(x)
        filter_maps = []
        for i, y in enumerate(x):
            y = y.reshape(self.n_channels, -1)
            filter_maps.append(self.lstm_autoencoders[i](y))
        x = torch.stack(filter_maps)
        x = self.conv_decoder(x)
        return x

    def train_model(self, model, train_dataset, val_dataset, train_index=None, test_index=None, N_EPOCHS=0, LEARNING_RATE=1e-3, MODEL_NAME='model', plot_result=True):

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #criterion = nn.L1Loss(reduction='sum').to(self.device)
        criterion = nn.MSELoss().to(self.device)
        history: Dict[str, Any] = dict(train=[], val=[], train_index=train_index, test_index=test_index)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 100000000000000.0

        for epoch in range(1, N_EPOCHS + 1):

            train_losses, val_losses = [], []

            begin_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {begin_time} -- Training Epoch {epoch}...', end=" ")

            model = model.train()

            for seq_true in train_dataset:
                optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model = model.eval()

            with torch.no_grad():
                for seq_true in val_dataset:
                    seq_true = seq_true.to(self.device)
                    seq_pred = model(seq_true)
                    loss = criterion(seq_pred, seq_true)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, str(MODEL_NAME+'.pth'))
                with open(str(MODEL_NAME+'.pickle'), 'wb') as handle:
                    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if plot_result:
                    plot_loss_throw_epochs(history, MODEL_NAME)

            end_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {end_time} -- train_loss: {round(train_loss, 6)}, val_loss: {round(val_loss, 6)}')

        model.load_state_dict(best_model_wts)

        return model.eval(), history
