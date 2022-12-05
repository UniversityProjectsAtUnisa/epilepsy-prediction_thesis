from utils.utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ENVIRONMENT VARIABLES
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS'))
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
H5_FILENAME = get_envvar("H5_FILENAME")
H5_DIRPATH = OUTPUT_PATH.joinpath(str(PREICTAL_SECONDS))
H5_FILEPATH = H5_DIRPATH.joinpath(H5_FILENAME)
SAVED_MODEL_DIR = get_envvar("SAVED_MODEL_DIR", "saves")
SAVED_MODEL_PATH = OUTPUT_PATH.joinpath(str(PREICTAL_SECONDS), SAVED_MODEL_DIR)
WINDOW_SIZE_SECONDS = int(get_envvar('WINDOW_SIZE_SECONDS'))
WINDOW_OVERLAP_SECONDS = int(get_envvar('WINDOW_OVERLAP_SECONDS'))
PATIENT_ID = get_envvar("PATIENT_ID", "")
METRICS_FILENAME = get_envvar("METRICS_FILENAME", "metrics.csv")
LOSS_PLOT_FILENAME = get_envvar("LOSS_PLOT_FILENAME", "loss_plot.png")
USE_CONVOLUTION = bool(get_envvar("USE_CONVOLUTION", ""))

CUDA_NAME = get_envvar("CUDA_NAME", "cuda")

RANDOM_STATE = int(get_envvar('RANDOM_STATE'))
N_SUBWINDOWS = int(get_envvar('N_SUBWINDOWS'))
ENCODING_DIM = int(get_envvar('ENCODING_DIM'))

LEARNING_RATE = float(get_envvar('LEARNING_RATE'))
N_EPOCHS = int(get_envvar('N_EPOCHS'))
BATCH_SIZE = int(get_envvar('BATCH_SIZE'))
PARTIAL_TRAINING = int(get_envvar('PARTIAL_TRAINING'))
PARTIAL_TESTING = int(get_envvar('PARTIAL_TRAINING'))

# CONSTANTS
METRIC_NAMES = ["seizures", "undetected%", "pred%", "spec%", "APT (s)", "mPT (s)", "MPT (s)", "PPV (%)", "BPPV (%)", "Imb. (%)"]
