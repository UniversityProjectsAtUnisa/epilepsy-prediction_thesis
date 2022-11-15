import os

# TODO: Duplicato
def get_envvar(name, default: str | None = None):
    """a function to get an environment variable that throws an exception if not found"""
    value = os.getenv(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable `{name}` not found")
