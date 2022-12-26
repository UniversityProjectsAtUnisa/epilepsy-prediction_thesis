from utils.utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ENVIRONMENT VARIABLES
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS'))
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
H5_FILENAME = get_envvar("H5_FILENAME")
H5_DIRPATH = OUTPUT_PATH
H5_FILEPATH = H5_DIRPATH.joinpath(H5_FILENAME)
SAVED_MODEL_DIR = get_envvar("SAVED_MODEL_DIR", "saves")
SAVED_MODEL_PATH = OUTPUT_PATH.joinpath(SAVED_MODEL_DIR)
WINDOW_SIZE_SECONDS = int(get_envvar('WINDOW_SIZE_SECONDS'))
WINDOW_OVERLAP_SECONDS = int(get_envvar('WINDOW_OVERLAP_SECONDS'))
PATIENT_ID = get_envvar("PATIENT_ID", "")
METRICS_FILENAME = get_envvar("METRICS_FILENAME", "metrics.csv")
LOSS_PLOT_FILENAME = get_envvar("LOSS_PLOT_FILENAME", "loss_plot.png")
CUMULATIVE_PREDICTIONS_FILENAME = get_envvar("CUMULATIVE_PREDICTIONS_FILENAME", "cum_preds.png")
USE_CONVOLUTION = bool(get_envvar("USE_CONVOLUTION", ""))

CUDA_NAME = get_envvar("CUDA_NAME", "cuda")

RANDOM_STATE = int(get_envvar('RANDOM_STATE'))
N_SUBWINDOWS = int(get_envvar('N_SUBWINDOWS'))

LEARNING_RATE = float(get_envvar('LEARNING_RATE'))
N_EPOCHS = int(get_envvar('N_EPOCHS'))
BATCH_SIZE = int(get_envvar('BATCH_SIZE'))
PARTIAL_TRAINING = int(get_envvar('PARTIAL_TRAINING'))
PARTIAL_TESTING = int(get_envvar('PARTIAL_TRAINING'))
SKIP_PATIENTS = [f'chb{int(id):02}' for id in get_envvar("SKIP_PATIENTS", "").split(",")]

# CONSTANTS
# ASD = Average Sequence Duration
# AIFP = Average Intra False Positive
METRIC_NAMES = ["train (s)", "val (s)", "test_normal (s)", "n_seizures", "ASD (s)", "pred%", "spec%",
                "IFP (s)", "APT (s)", "mPT (s)", "MPT (s)", "PPV (%)", "BPPV (%)", "Imb. (%)"]
# METRIC_NAMES = ["normal_files"] + BASIC_METRICS
# METRIC_NAMES = FOLDMETRIC_NAMES
