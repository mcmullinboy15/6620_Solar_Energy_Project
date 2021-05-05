import os
import sys

import numpy as np
import pandas as pd
from model_and_defs import separate_train_val_test

directory = 'data/'

for filename in os.listdir(directory):
    filename = f"{directory}{filename}"
    data = pd.read_csv(filename, header=None)
    data = separate_train_val_test(data)
    for phase in ['train', 'val', 'test']:
        y = data[phase]['y']
        x = data[phase]['x']
        phase_data = np.c_[y, x]
        phase_data = pd.DataFrame(phase_data)
        phase_data.to_csv(f"{filename[:-4]}-{phase}.csv", index=False, header=False)
