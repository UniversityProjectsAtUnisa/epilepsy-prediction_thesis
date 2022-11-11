import os
import h5py
import numpy as np
import typing


def get_envvar(name, default: str | None = None):
    """a function to get an environment variable that throws an exception if not found"""
    value = os.getenv(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable `{name}` not found")


class ResizableH5Dataset:
    def __init__(self, h5_path: str, compression='lzf'):
        self.h5_path = h5_path
        self.compression = compression
        self._created = False

    def append_data(self, h5_file: h5py.File, data: np.ndarray):
        if not self._created:
            sample_shape = data.shape[1:]
            kwargs = dict(maxshape=(None, *sample_shape), dtype=data.dtype, compression=self.compression)
            ds = h5_file.create_dataset(self.h5_path, shape=data.shape, **kwargs)
            ds[:] = data
            self._created = True
        else:
            ds = typing.cast(h5py.Dataset, h5_file[self.h5_path])
            curr_size = ds.shape[0]
            ds.resize(curr_size + data.shape[0], axis=0)
            ds[curr_size:] = data
