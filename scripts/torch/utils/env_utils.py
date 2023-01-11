import os
from typing import Optional


def get_envvar(name, default: Optional[str] = None):
    """a function to get an environment variable that throws an exception if not found"""
    value = os.getenv(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable `{name}` not found")


def get_bool_envvar(name, default: Optional[bool] = None):
    if default is None:
        default = False
    value = get_envvar(name, str(default))
    return value.lower() in ("yes", "true", "t", "1")
