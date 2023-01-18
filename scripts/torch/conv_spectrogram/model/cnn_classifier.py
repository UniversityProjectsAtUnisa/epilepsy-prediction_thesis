from .modules.feature_extractor import FeatureExtractor
from .modules.dense_classifier import DenseClassifier

import torch
from ...helpers.history import History
import copy
import math
from torch import nn
import pathlib
import time
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Optional
from torch.utils.data import Dataset
from ...utils.gpu_utils import device_context
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score


class WindowDataset(Dataset):
    def __init__(self, X, y):
        if len(X) != len(y):
            raise Exception("X and y must have the same length")
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


def build_sampler(y: np.ndarray):
    false_probability = sum(y == 0)/len(y)
    weights = [false_probability, 1-false_probability]
    samples_weight = np.where(y == 0, weights[1], weights[0])
    return WeightedRandomSampler(samples_weight, len(samples_weight))


class CNNClassifier(nn.Module):
    fe_filename = "feature_extractor.pth"
    mlp_filename = "dense_classifier.pth"
    model_dirname = "cnnclassifier"
    history_filename = "history.pkl"
    checkpoint_filename = "checkpoint.pth"

    def __init__(self, fe: Optional[FeatureExtractor] = None, mlp: Optional[DenseClassifier] = None):
        super(CNNClassifier, self).__init__()
        self.fe = fe if fe is not None else FeatureExtractor()
        self.mlp = mlp if mlp is not None else DenseClassifier()

    def forward(self, x):
        if self.fe is None or self.mlp is None:
            raise Exception("Model not initialized")
        return self.mlp(self.fe(x))

    @ classmethod
    def load(cls, dirpath: pathlib.Path):
        fe = FeatureExtractor.load(dirpath/cls.fe_filename)
        mlp = DenseClassifier.load(dirpath/cls.mlp_filename)
        return cls(fe, mlp)

    def save(self, dirpath: pathlib.Path):
        if self.fe is None or self.mlp is None:
            raise Exception("Model not initialized")
        self.fe.save(dirpath/self.fe_filename)
        self.mlp.save(dirpath/self.mlp_filename)

    def load_checkpoint(self, dirpath: pathlib.Path) -> History:
        model_dirpath = dirpath/self.model_dirname

        history = History.load(model_dirpath/self.history_filename)

        checkpoint = torch.load(model_dirpath/self.checkpoint_filename)
        self.load_state_dict(checkpoint)

        return history

    def save_checkpoint(self, dirpath: pathlib.Path, history: History):
        model_dirpath = dirpath/self.model_dirname
        model_dirpath.mkdir(parents=True, exist_ok=True)

        history.save(model_dirpath/self.history_filename)

        torch.save(self.state_dict(), model_dirpath/self.checkpoint_filename)

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        was_on_cpu = not X.is_cuda
        X = X.to(device_context.device)
        with torch.no_grad():
            self.eval()
            y_proba = self(X)
        if was_on_cpu:
            y_proba = y_proba.cpu()
        return y_proba

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs, batch_size=64, dirpath: pathlib.Path = pathlib.Path("/tmp"), learning_rate=1e-4, patience=15):
        early_stopping_counter = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        history = History()

        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = math.inf

        y_train_np = y_train.cpu().numpy()
        y_val_np = y_val.cpu().numpy()

        train_dataset = WindowDataset(X_train, y_train)
        val_dataset = WindowDataset(X_val, y_val)
        sampler = build_sampler(y_train_np)

        start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(f"Training started: {start_time}")
        for epoch in range(1, n_epochs + 1):

            train_loss = 0
            val_losses = []
            train_preds, val_preds = [], []

            self.train()

            for X, y in DataLoader(train_dataset, batch_size=batch_size, pin_memory=not X_train.is_cuda, generator=torch.Generator(
                    device=device_context.device), sampler=sampler):
                X = X.to(device_context.device)
                y = y.to(device_context.device)
                optimizer.zero_grad()
                pred = self(X).flatten()
                loss = criterion(pred, y)
                train_preds.extend((pred > 0.5).tolist())
                loss.backward()
                optimizer.step()
                train_loss = loss.item()

            self.eval()

            with torch.no_grad():
                for X, y in DataLoader(val_dataset, batch_size=4096, pin_memory=not X_train.is_cuda, generator=torch.Generator(
                        device=device_context.device)):
                    X = X.to(device_context.device)
                    y = y.to(device_context.device)
                    pred = self(X).flatten()
                    loss = criterion(pred, y)
                    val_preds.extend((pred > 0.5).tolist())
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            history.add(train_loss, val_loss)

            # print current time in format HH:MM:SS
            end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print(
                f"""{epoch}/{n_epochs}, time: {end_time}, 
                train_loss: {train_loss:.4f}, 
                val_loss: {val_loss:.4f}, 
                train_acc: {accuracy_score(y_train_np, train_preds):.4f}, 
                val_acc: {accuracy_score(y_val_np, val_preds):.4f},
                balanced_train_acc: {balanced_accuracy_score(y_train_np, train_preds):.4f},
                balanced_val_acc: {balanced_accuracy_score(y_val_np, val_preds):.4f},
                f1_train: {f1_score(y_train_np, train_preds):.4f},
                f1_val: {f1_score(y_val_np, val_preds):.4f}
                precision_train: {precision_score(y_train_np, train_preds):.4f},
                precision_val: {precision_score(y_val_np, val_preds):.4f}
                recall_train: {recall_score(y_train_np, train_preds):.4f},
                recall_val: {recall_score(y_val_np, val_preds):.4f}
                """)

            if val_loss < best_loss:
                early_stopping_counter = 0
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                self.save_checkpoint(dirpath, history)

            else:
                early_stopping_counter += 1
                if early_stopping_counter == patience:
                    print("Early stopping")
                    break

        self.load_state_dict(best_model_wts)
        return history
