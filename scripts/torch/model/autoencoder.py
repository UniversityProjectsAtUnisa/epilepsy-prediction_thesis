from torch import nn
from typing import Dict, Any
import torch
import time
import copy
import numpy as np
import pickle
from .modules.conv_encoder import ConvEncoder
from .modules.conv_decoder import ConvDecoder
from .modules.lstm_autoencoder import LSTMAutoencoder
from .modules.standardizer import Standardizer
from .helpers import plotfunction as pf
from torch.utils.data import DataLoader


def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', index=0)
    print(f'(Using device: {device})', end=" ")
    return device


class Autoencoder(nn.Module):

    def __init__(self, sample_length, n_filters, n_channels, kernel_size, n_subwindows):
        super(Autoencoder, self).__init__()
        self.n_subwindows = n_subwindows
        n_channels = n_channels

        self.device = check_device()

        len_subwindows = sample_length//n_subwindows
        encoding_dim = len_subwindows//2

        self.conv_encoder = ConvEncoder(sample_length=sample_length, n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size).to(self.device)
        self.conv_decoder = ConvDecoder(sample_length=sample_length, n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size).to(self.device)
        self.lstm_autoencoders = [LSTMAutoencoder(seq_len=n_subwindows, n_features=len_subwindows,
                                                  encoding_dim=encoding_dim).to(self.device) for _ in range(n_filters)]

        self.standardizers = [Standardizer() for _ in range(n_channels)]  # Da spostare

    def forward(self, x):
        x = self.conv_encoder(x)
        filter_maps = []
        for i in range(x.shape[1]):
            y = x[:, i, :, :]
            y = y.reshape(y.shape[0], self.n_subwindows, -1)
            y = self.lstm_autoencoders[i](y)
            y = y.reshape(y.shape[0], 1, -1)
            filter_maps.append(y)
        x = torch.stack(filter_maps, dim=1)
        x = self.conv_decoder(x)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def train_model(self, model, train_dataset, val_dataset, train_index=None, test_index=None, n_epochs=0, learning_rate=1e-3, model_name='model', plot_result=False):

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss().to(self.device)
        history: Dict[str, Any] = dict(train=[], val=[], train_index=train_index, test_index=test_index)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 100000000000000.0

        for epoch in range(1, n_epochs + 1):

            train_losses, val_losses = [], []

            begin_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {begin_time} -- Training Epoch {epoch}...', end=" ")

            model = model.train()

            # for seq_true in train_dataset:
            for seq_true in DataLoader(train_dataset, batch_size=32, shuffle=True):
                optimizer.zero_grad()
                seq_true = seq_true.to(self.device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model = model.eval()

            with torch.no_grad():
                # for seq_true in val_dataset:
                for seq_true in val_dataset:
                    seq_true = val_dataset
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
                torch.save(best_model_wts, str(model_name+'.pth'))
                with open(str(model_name+'.pickle'), 'wb') as handle:
                    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if plot_result:
                    pf.plot_loss_throw_epochs(history, model_name)

            end_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {end_time} -- train_loss: {round(train_loss, 6)}, val_loss: {round(val_loss, 6)}')

        model.load_state_dict(best_model_wts)

        return model.eval(), history
