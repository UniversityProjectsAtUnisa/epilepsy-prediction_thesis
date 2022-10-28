from utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = pathlib.Path(get_envvar('DATASET_PATH'))
PREICTAL_SECONDS = int(get_envvar("PREICTAL_SECONDS"))
POSTICTAL_SECONDS = int(get_envvar("POSTICTAL_SECONDS"))
SAMPLING_FREQUENCY = 256
