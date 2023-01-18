import numpy as np
import h5py
import pathlib


def preprocess_data(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=1)


def load_x_data_new(dataset_path, key, preprocess, n_subwindows=2):
    x_data = []
    with h5py.File(dataset_path) as f:
        for x in f[key].values():  # type: ignore
            x = x[:]
            if preprocess:
                x = preprocess_data(x)
            x_data.append(x)
    return tuple(x_data)


def f(X: np.ndarray):
    # X has shape (5,2,4,3). Output has shape (5,4,3,2).
    # this function executes np.array([np.concatenate(x, -1) for x in X]) but faster
    return np.moveaxis(np.concatenate(X, -1), 1, -1)


def segment_data(X: np.ndarray, n_subwindows: int, overlap: int) -> np.ndarray:
    stride = n_subwindows - overlap
    skip = (len(X) - overlap) % stride

    segments = np.array(tuple(zip(*(X[skip+i::stride] for i in range(n_subwindows)))))
    return np.moveaxis(segments, 1, -1)


if __name__ == "__main__":
    arr = np.arange(10*4*3).reshape(10, 4, 3)
    window_size = 5
    overlap = 3
    res = segment_data(arr, window_size, overlap)
    print(res)

    dataset_path = pathlib.Path("/media/marco741/Archive/Datasets/spectrograms/dataset.h5")
    res1 = load_x_data_new(dataset_path, "chb01/train/X", False)
    res2 = tuple(segment_data(x, 6, 5) for x in res1)
    print(res2)
