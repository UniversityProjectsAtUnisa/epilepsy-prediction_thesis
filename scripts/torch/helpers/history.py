import pickle
import pathlib


class History:
    def __init__(self):
        self.train = []
        self.val = []

    def add(self, train_loss, val_loss):
        self.train.append(train_loss)
        self.val.append(val_loss)

    def save(self, path: pathlib.Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: pathlib.Path):
        with open(path, 'rb') as f:
            history = pickle.load(f)
        if not isinstance(history, cls):
            raise Exception("Invalid history file")
        return history
