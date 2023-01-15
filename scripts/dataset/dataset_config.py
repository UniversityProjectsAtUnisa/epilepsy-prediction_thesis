import pathlib

from dotenv import load_dotenv
from utils import get_envvar, get_bool_envvar

load_dotenv()

# ==MANDATORY
DATASET_DIRPATH = pathlib.Path(get_envvar('DATASET_DIRPATH'))
OUTPUT_DIRPATH = pathlib.Path(get_envvar('OUTPUT_DIRPATH'))
WINDOW_SIZE_SECONDS = int(get_envvar('WINDOW_SIZE_SECONDS'))
WINDOW_OVERLAP_SECONDS = int(get_envvar('WINDOW_OVERLAP_SECONDS'))

# ==OPTIONAL: PATH
H5_FILEPATH = get_envvar("H5_FILEPATH", "")
LABELS_DIRPATH = pathlib.Path(get_envvar('LABELS_DIRPATH', "scripts/dataset/labels"))

# ==OPTIONAL: DEFAULT FILENAMES
H5_FILENAME = get_envvar("H5_FILENAME", "dataset.h5")
NORMAL_LABELS_FILENAME = get_envvar("NORMAL_LABELS_FILENAME", "normal_labels.npy")
SEIZURE_LABELS_FILENAME = get_envvar("SEIZURE_LABELS_FILENAME", "seizure_labels.npy")

# ==OPTIONAL: DATASET
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS', "300"))
POSTICTAL_SECONDS = int(get_envvar('POSTICTAL_SECONDS', "1000"))
USE_SPECTROGRAMS = get_bool_envvar("USE_SPECTROGRAMS", False)

# =DERIVED
H5_FILEPATH = OUTPUT_DIRPATH/H5_FILENAME if H5_FILEPATH == "" else pathlib.Path(H5_FILEPATH)
H5_DIRPATH = H5_FILEPATH.parent
NORMAL_LABELS_FILEPATH = LABELS_DIRPATH/NORMAL_LABELS_FILENAME
SEIZURE_LABELS_FILEPATH = LABELS_DIRPATH/SEIZURE_LABELS_FILENAME

# =CONSTANTS
SAMPLING_FREQUENCY = 256
USEFUL_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8"
]
