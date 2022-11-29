from matplotlib import pyplot as plt


def plot_train_val_losses(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(len(train_losses)))
    ax.plot(epochs, train_losses, label="train_loss", color="blue")
    ax.plot(epochs, val_losses, label="val_loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig
