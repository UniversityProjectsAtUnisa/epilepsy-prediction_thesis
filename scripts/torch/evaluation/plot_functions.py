from matplotlib import pyplot as plt
import numpy as np
import pathlib


def plot_history(history, dirpath: pathlib.Path):
    history_plot = plot_train_val_losses(history.train, history.val)
    history_plot.savefig(str(dirpath))
    plt.close(history_plot)


def plot_train_val_losses(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(len(train_losses)))
    ax.plot(epochs, train_losses, label="train_loss", color="blue")
    ax.plot(epochs, val_losses, label="val_loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig


def plot_cumulative_positive_preds(fold_preds, window_size_seconds, window_overlap_seconds):
    avg_preds = []
    for i in range(len(fold_preds[0])):
        avg = np.array([p[i].numpy() for p in fold_preds]).mean(axis=0)
        avg_preds.append(avg)
    # equalslength_preds = [np.pad(pred, (max_length - len(pred), 0), "constant") for pred in avg_preds]
    cum_preds = [np.insert(fp.cumsum(), 0, 0) for fp in avg_preds]
    max_length = max(len(pred) for pred in cum_preds)
    scaled_preds = [pred / (max_length - 1) for pred in cum_preds]
    fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(np.linspace(0, 10, 11), np.linspace(0, 1, 11), label="reference", color="red")
    for pred in scaled_preds:
        ax.plot(np.linspace(-max_length * (window_size_seconds - window_overlap_seconds), 0, max_length, endpoint=False)[max_length - len(pred):], pred)
    ax.set_xlabel("Seconds relative to seizure (s)")
    ax.set_ylabel("Cumulative positive predictions (%)")
    # ax.legend()
    return fig


def plot_cumulative_negative_preds(fold_preds, window_size_seconds, window_overlap_seconds):
    preds = [p.numpy() for p in fold_preds]
    cum_preds = [np.insert(pred.cumsum(), 0, 0) for pred in preds]
    scaled_preds = [pred / (len(pred) - 1) for pred in cum_preds]

    fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(np.linspace(0, 10, 11), np.linspace(0, 1, 11), label="reference", color="red")
    for pred in scaled_preds:
        ax.plot(np.linspace(0, 1, len(pred)), pred)
    ax.set_xlabel("Normal data duration (%)")
    ax.set_ylabel("Cumulative positive predictions (%)")
    # ax.legend()
    return fig
