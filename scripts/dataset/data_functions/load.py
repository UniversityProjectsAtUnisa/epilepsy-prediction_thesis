import numpy as np
import pandas as pd
import pathlib
from typing import List


def load_paper_labels(filepath: pathlib.Path):
    data_y = np.load(filepath, allow_pickle=True)  # metadatos
    data_y = pd.DataFrame(data_y, columns=['type', 'name', 'filename', 'pre1', 'pre2', 'id_eeg_actual', 'id_eeg_all', 'label'])
    data_y.label = data_y.label.astype('int')
    return data_y
