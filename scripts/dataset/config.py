from utils import get_envvar
import pathlib
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = pathlib.Path(get_envvar('DATASET_PATH'))
