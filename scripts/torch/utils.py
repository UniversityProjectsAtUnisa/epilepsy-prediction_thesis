import os
import torch


def get_envvar(name, default: str | None = None):
    # TODO: Duplicato
    """a function to get an environment variable that throws an exception if not found"""
    value = os.getenv(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable `{name}` not found")


def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', index=0)
    print(f'(Using device: {device})', end=" ")
    return device
