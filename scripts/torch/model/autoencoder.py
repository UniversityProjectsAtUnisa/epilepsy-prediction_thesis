import copy
import math
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.gpu_utils import device_context

from .helpers import plotfunction as pf
from .helpers.history import History
from .modules.conv_decoder import ConvDecoder
from .modules.conv_encoder import ConvEncoder
from .modules.lstm_autoencoder import LSTMAutoencoder


class Autoencoder(nn.Module):
    history_filename = 'history.pkl'
    model_filename = 'model.pth'
    state_dict_filename = 'checkpoint.pth'

    def __init__(self, sample_length, n_subwindows):
        super(Autoencoder, self).__init__()
        self.n_subwindows = n_subwindows

        len_subwindows = sample_length//n_subwindows
        encoding_dim = len_subwindows//2
        self.lstm_autoencoder = LSTMAutoencoder(seq_len=n_subwindows, n_features=len_subwindows, encoding_dim=encoding_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.n_subwindows, -1)
        x = self.lstm_autoencoder(x)
        x = x.reshape(x.shape[0], -1)
        return x

    def predict(self, X):
        predictions, _ = self.predict_with_losses(X)
        return predictions

    def calculate_losses(self, X):
        _, losses = self.predict_with_losses(X)
        return losses

    def predict_with_losses(self, X_true):
        # criterion = nn.L1Loss(reduction='sum').to(model.device)
        was_on_cpu = not X_true.is_cuda
        X_true = X_true.to(device_context.device)
        criterion = nn.MSELoss(reduction="none")
        with torch.no_grad():
            self.eval()
            X_pred = self(X_true)
            dim = tuple(range(1, len(X_true.shape)))
            losses = criterion(X_pred, X_true).mean(dim)
        if was_on_cpu:
            X_pred = X_pred.cpu()
            losses = losses.cpu()
        return X_pred, losses

    def train_model(self, X_train, X_val, n_epochs, batch_size=64, dirpath: pathlib.Path = pathlib.Path("/tmp"), learning_rate=1e-3, plot_result=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        history = History()

        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = math.inf

        start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(f"Training started: {start_time}")
        for epoch in range(1, n_epochs + 1):

            train_losses, val_losses = [], []

            self.train()

            for seq_true in DataLoader(X_train, batch_size=batch_size, shuffle=True, pin_memory=not X_train.is_cuda, generator=torch.Generator(
                    device=device_context.device)):
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

            # print current time in format HH:MM:SS
            end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print(f'{epoch}/{n_epochs}, time: {end_time}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

        self.load_state_dict(best_model_wts)
        return history

    @classmethod
    def load(cls, dirpath: pathlib.Path):
        model_path = dirpath.joinpath(cls.model_filename)
        model = torch.load(model_path, map_location=device_context.device)
        if not isinstance(model, cls):
            raise Exception("Invalid model file")
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
