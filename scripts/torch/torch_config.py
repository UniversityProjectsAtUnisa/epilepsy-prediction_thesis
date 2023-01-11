from utils.env_utils import get_envvar, get_bool_envvar
import pathlib
from dotenv import load_dotenv
from utils.types import FinetuningMode

load_dotenv()

# ==MANDATORY
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
WINDOW_SIZE_SECONDS = int(get_envvar('WINDOW_SIZE_SECONDS'))
WINDOW_OVERLAP_SECONDS = int(get_envvar('WINDOW_OVERLAP_SECONDS'))

# ==OPTIONAL
H5_FILEPATH = get_envvar("H5_FILEPATH", "")

# ==OPTIONAL: PATH
PATIENT_GENERIC_OUTPUT_DIRNAME = get_envvar("PATIENT_GENERIC_OUTPUT_PATH", "patientgeneric")
H5_FILENAME = get_envvar("H5_FILENAME", "dataset.h5")
SAVED_MODEL_DIRNAME = get_envvar("SAVED_MODEL_DIRNAME", "saves")
METRICS_FILENAME = get_envvar("METRICS_FILENAME", "metrics.csv")
LOSS_PLOT_FILENAME = get_envvar("LOSS_PLOT_FILENAME", "loss_plot.png")
CUMULATIVE_PREDICTIONS_FILENAME = get_envvar("CUMULATIVE_PREDICTIONS_FILENAME", "cum_preds.png")

# ==OPTIONAL: MODEL
N_SUBWINDOWS = int(get_envvar('N_SUBWINDOWS', "2"))

# ==OPTIONAL: TRAINING
LEARNING_RATE = float(get_envvar('LEARNING_RATE', "1e-3"))
N_EPOCHS = int(get_envvar('N_EPOCHS', "150"))
BATCH_SIZE = int(get_envvar('BATCH_SIZE', "32"))
FINETUNING_MODE = FinetuningMode(get_envvar("FINETUNING_MODE", "full").lower())

# ==OPTIONAL: PERFORMANCE
PARALLEL_TRAINING = get_bool_envvar("PARALLEL_TRAINING", False)
PARALLEL_WORKERS = int(get_envvar('PARALLEL_WORKERS', "0"))
CUDA_NAME = get_envvar("CUDA_NAME", "cuda")
SKIP_PATIENTS = [f'chb{int(id):02}' for id in get_envvar("SKIP_PATIENTS", "").split(",") if id != ""]

# ==OTHERS
RANDOM_STATE = int(get_envvar('RANDOM_STATE', "1"))
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS', "300"))


# ==DERIVED
H5_FILEPATH = OUTPUT_PATH/H5_FILEPATH if H5_FILEPATH == "" else pathlib.Path(H5_FILEPATH)
H5_DIRPATH = H5_FILEPATH.parent
PATIENT_GENERIC_OUTPUT_DIRPATH = OUTPUT_PATH/PATIENT_GENERIC_OUTPUT_DIRNAME
SAVED_MODEL_PATH = OUTPUT_PATH/SAVED_MODEL_DIRNAME


# =CONSTANTS
# ASD is Average Sequence Duration
# IFP is Inter False Positive
METRIC_NAMES = ["train (s)", "val (s)", "test_normal (s)", "n_seizures", "ASD (s)", "pred%", "spec%",
                "IFP (s)", "APT (s)", "mPT (s)", "MPT (s)", "PPV (%)", "BPPV (%)", "Imb. (%)"]
# METRIC_NAMES = ["normal_files"] + BASIC_METRICS
# METRIC_NAMES = FOLDMETRIC_NAMES
