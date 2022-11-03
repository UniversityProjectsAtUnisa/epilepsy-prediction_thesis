from utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ENVIRONMENT VARIABLES
DATASET_PATH = pathlib.Path(get_envvar('DATASET_PATH'))
PREICTAL_SECONDS = int(get_envvar('PREICTAL_SECONDS'))
POSTICTAL_SECONDS = int(get_envvar('POSTICTAL_SECONDS'))
OUTPUT_PATH = pathlib.Path(get_envvar('OUTPUT_PATH'))
FORBIDDEN_PATHNAMES = pathlib.Path(get_envvar('FORBIDDEN_PATHNAMES_PATH')).read_text().splitlines()

_PARTIAL_PATHNAMES = pathlib.Path(get_envvar('PARTIAL_PATHNAMES_PATH')).read_text().splitlines()
PARTIAL_PATHNAMES = {}
for line in _PARTIAL_PATHNAMES:
    name, channels = line.split(':')
    channels = channels.split(',')
    PARTIAL_PATHNAMES[name] = list(map(int, channels))

# CONSTANTS
SAMPLING_FREQUENCY = 256
