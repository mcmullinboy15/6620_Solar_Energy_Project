import os
import torch
import pandas as pd
from Train import Train
from model_and_defs import Model2
from loop import loop_train


testing = True


live_params = {
    'EPOCHS'  : [2 ** i for i in range(8, 13)],
    'BATCHES' : [2 ** i for i in range(4, 9)],
    'MODELS'  : [Model2],
    'H1S'     : [64, 128, 256],
    'H2S'     : [64, 128, 256],
    'H3S'     : [64, 128, 256],
    'LRS'     : [1e-4],
    'OPTIMS'  : [torch.optim.Adam],
}
test_params = {
    'EPOCHS'  : [200],
    'BATCHES' : [64],
    'MODELS'  : [Model2],
    'H1S'     : [128],
    'H2S'     : [128],
    'H3S'     : [128],
    'LRS'     : [1e-4],
    'OPTIMS'  : [torch.optim.Adam],
}

params = test_params if testing else live_params

trained = []
lowests = []

data_folder = f"data/"


data_files = os.listdir(data_folder)  # if not testing else os.listdir(data_folder)[:1]
for data_filename in data_files:
    data_path = f"{data_folder}{data_filename}"

    if data_filename.__contains__("norm") or data_filename.__contains__("tight") or data_filename.__contains__("none"):  # or data_filename.__contains__("solar") or data_filename.__contains__("percentage"):
        continue

    ss_scaled = data_path.__contains__('ss')
    ss = '-ss' if ss_scaled else ''

    # I'm going to skip the ss ones
    if not ss_scaled:
        trained.append(
            loop_train(f'models/', f'{data_filename[:-4]}{ss}', params, data_path, ss_scaled, skip_done=False, plot_train=True)
        )

lowest = 1000000000000
for traine in trained:
    for train in traine:
        if train.lowest_val_loss < lowest:
            lowest = train.lowest_val_loss
            lowests.append(train)

for low in lowests:
    print('lowest:', low.filename, low.lowest_val_loss)
