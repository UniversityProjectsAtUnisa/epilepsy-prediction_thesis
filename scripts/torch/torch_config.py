from utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ENVIRONMENT VARIABLES
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS'))
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
H5_FILENAME = get_envvar("H5_FILENAME")
H5_PATH = OUTPUT_PATH.joinpath(str(PREICTAL_SECONDS), H5_FILENAME)
SAVED_MODEL_PATH = OUTPUT_PATH.joinpath(str(PREICTAL_SECONDS), "saves")

# CONSTANTS
N_FILTERS = int(get_envvar('N_FILTERS'))
KERNEL_SIZE = int(get_envvar('KERNEL_SIZE'))
N_SUBWINDOWS = int(get_envvar('N_SUBWINDOWS'))

BATCH_SIZE = int(get_envvar('BATCH_SIZE'))
PARTIAL_TRAINING = int(get_envvar('PARTIAL_TRAINING'))
