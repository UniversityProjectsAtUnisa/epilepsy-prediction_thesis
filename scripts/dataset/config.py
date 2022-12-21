import pathlib

from dotenv import load_dotenv
from utils import get_envvar

load_dotenv()

# ENVIRONMENT VARIABLES
DATASET_PATH = pathlib.Path(get_envvar('DATASET_PATH'))
NUMPY_DATASET_PATH = pathlib.Path(get_envvar('NUMPY_DATASET_PATH'))
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS'))
POSTICTAL_SECONDS = int(get_envvar('POSTICTAL_SECONDS'))
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
FORBIDDEN_PATHNAMES = pathlib.Path(get_envvar('FORBIDDEN_PATHNAMES_PATH')).read_text().splitlines()
DISCARDED_EDFS = pathlib.Path(get_envvar('DISCARDED_EDFS_PATH')).read_text().splitlines()
SLICES_FILENAME = get_envvar("SLICES_FILENAME")
SLICES_ANALYSIS_FILENAME = get_envvar("SLICES_ANALYSIS_FILENAME")
H5_FILENAME = get_envvar("H5_FILENAME")
H5_DIRPATH = OUTPUT_PATH.joinpath(str(PREICTAL_SECONDS))
H5_FILEPATH = H5_DIRPATH.joinpath(H5_FILENAME)
USEFUL_CHANNELS = pathlib.Path(get_envvar("USEFUL_CHANNELS_FILENAME")).read_text().splitlines()
WINDOW_SIZE_SECONDS = int(get_envvar('WINDOW_SIZE_SECONDS'))
WINDOW_OVERLAP_SECONDS = int(get_envvar('WINDOW_OVERLAP_SECONDS'))
USE_SPECTROGRAMS = bool(get_envvar("USE_SPECTROGRAMS", ""))


_PARTIAL_PATHNAMES = pathlib.Path(get_envvar('PARTIAL_PATHNAMES_PATH')).read_text().splitlines()
PARTIAL_PATHNAMES = {}
for line in _PARTIAL_PATHNAMES:
    name, channels = line.split(':')
    channels = channels.split(',')
    PARTIAL_PATHNAMES[name] = list(map(int, channels))

# CONSTANTS
SAMPLING_FREQUENCY = 256
