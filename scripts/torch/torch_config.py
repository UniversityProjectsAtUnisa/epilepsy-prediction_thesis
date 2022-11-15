from utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ENVIRONMENT VARIABLES
DATASET_PATH = pathlib.Path(get_envvar('DATASET_PATH'))
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS'))
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
H5_FILENAME = get_envvar("H5_FILENAME")
H5_PATH = OUTPUT_PATH.joinpath(str(PREICTAL_SECONDS), H5_FILENAME)

# CONSTANTS
N_FILTERS = 3
KERNEL_SIZE = 3
N_SUBWINDOWS = 12
