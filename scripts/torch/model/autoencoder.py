from torch import nn
import torch
import time
import copy
import numpy as np
from .modules.conv_encoder import ConvEncoder
from .modules.conv_decoder import ConvDecoder
from .modules.lstm_autoencoder import LSTMAutoencoder
from .helpers import plotfunction as pf
from .helpers.history import History
from torch.utils.data import DataLoader
import math
import pathlib
import torch_config as config
from utils.gpu_utils import device_context


class Autoencoder(nn.Module):
    history_filename = 'history.pkl'
    model_filename = 'model.pth'
    state_dict_filename = 'checkpoint.pth'

    def __init__(self, sample_length, n_filters, n_channels, kernel_size, n_subwindows):
        super(Autoencoder, self).__init__()
        self.n_subwindows = n_subwindows
        n_channels = n_channels

        len_subwindows = sample_length//n_subwindows
        encoding_dim = len_subwindows//2

        self.conv_encoder = ConvEncoder(sample_length=sample_length, n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size)
        self.conv_decoder = ConvDecoder(sample_length=sample_length, n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size)
        for i in range(n_filters):
            setattr(self, f"lstm_autoencoder_{i}", LSTMAutoencoder(seq_len=n_subwindows, n_features=len_subwindows, encoding_dim=encoding_dim))

    def forward(self, x):
        x = self.conv_encoder(x)

        filter_maps = []
        for i in range(x.shape[1]):
            y = x[:, i, :, :]
            y = y.reshape(y.shape[0], self.n_subwindows, -1)
            y = getattr(self, f"lstm_autoencoder_{i}")(y)
            y = y.reshape(y.shape[0], 1, -1)
            filter_maps.append(y)
        x = torch.stack(filter_maps, dim=1)
        x = self.conv_decoder(x)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def predict(self, X):
        predictions, _ = self.predict_with_losses(X)
        return predictions

    def calculate_losses(self, X):
        _, losses = self.predict_with_losses(X)
        return losses

    def predict_with_losses(self, X_true):
        # criterion = nn.L1Loss(reduction='sum').to(model.device)
        X_true = X_true.to(device_context.device)
        criterion = nn.MSELoss(reduction="none")
        with torch.no_grad():
            self.eval()
            X_pred = self(X_true)
            dim = tuple(range(1, len(X_true.shape)))
            losses = criterion(X_pred, X_true).mean(dim)
        return X_pred, losses

    def train_model(self, X_train, X_val, n_epochs, batch_size=64, dirpath: pathlib.Path = pathlib.Path("/tmp"), learning_rate=1e-3, plot_result=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        history = History()

        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = math.inf

        for epoch in range(1, n_epochs + 1):

            train_losses, val_losses = [], []

            begin_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {begin_time} -- Training Epoch {epoch}...', end=" ")

            # self = self.train()
            self.train()

            for seq_true in DataLoader(X_train, batch_size=batch_size, shuffle=True, pin_memory=True, generator=torch.Generator(device=device_context.device)):
                seq_true = seq_true.to(device_context.device)
                optimizer.zero_grad()
                seq_pred = self(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.eval()

            with torch.no_grad():
                seq_true = X_val.to(device_context.device)
                seq_pred = self(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history.add(train_loss, val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                self.save_checkpoint(dirpath, history)
                if plot_result:
                    pf.plot_loss_throw_epochs(history, "model")

            end_time = time.strftime("%H:%M:%S", time.localtime())
            print(f'Time: {end_time} -- train_loss: {round(train_loss, 6)}, val_loss: {round(val_loss, 6)}')

        self.load_state_dict(best_model_wts)
        return history

    @classmethod
    def load(cls, dirpath: pathlib.Path):
        model_path = dirpath.joinpath(cls.model_filename)
        model = torch.load(model_path)
        if not isinstance(model, cls):
            raise Exception("Invalid model file")
        return model

    @classmethod
    def load_from_checkpoint(cls, dirpath: pathlib.Path, sample_length, n_filters, n_channels, kernel_size, n_subwindows):
        model = cls(sample_length, n_filters, n_channels, kernel_size, n_subwindows)
        model.load_checkpoint(dirpath)
        return model

    def save(self, dirpath: pathlib.Path):
        dirpath.mkdir(parents=True, exist_ok=True)
        model_path = dirpath.joinpath(self.model_filename)
        torch.save(self, model_path)

    def load_checkpoint(self, dirpath: pathlib.Path) -> History:
        history_path = dirpath.joinpath(self.history_filename)
        history = History.load(history_path)

        state_dict_path = dirpath.joinpath(self.state_dict_filename)
        self.load_state_dict(torch.load(state_dict_path))

        return history

    def save_checkpoint(self, dirpath: pathlib.Path, history: History):
        dirpath.mkdir(parents=True, exist_ok=True)
        history_path = dirpath.joinpath(self.history_filename)
        history.save(history_path)

        state_dict_path = dirpath.joinpath(self.state_dict_filename)
        torch.save(self.state_dict(), state_dict_path)
