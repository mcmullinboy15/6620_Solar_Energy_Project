import os
import shutil

import torch

counter = 0
for batch in [2 ** i for i in range(4, 9)]:
    sum = 0
    lowest_model = None
    data_folder = f"../models/{batch}/"

    for data_filename in os.listdir(data_folder):

        if not data_filename.endswith(".pth"):
            continue

        data_path = f"{data_folder}{data_filename}"
        try:
            loaded_dict = torch.load(data_path, map_location=torch.device('cpu'))
        except RuntimeError as e:
            print(e)
            continue

        lowest_val_loss = loaded_dict['lowest_val_loss']

        if lowest_model is None:
            lowest_model = (loaded_dict, data_path)

        if lowest_val_loss < lowest_model[0]["lowest_val_loss"]:
            lowest_model = (loaded_dict, data_path)

        if lowest_val_loss < 0.20:
            sum += lowest_val_loss
            # print(data_path, lowest_val_loss)

            # get and convert to Andrews_v2_*th for the Wet mill computer
            # shutil.copy(data_path, f"../models/best/{counter}-{data_filename}")
            counter += 1
    print(f"Batch ({batch}) sum:", sum)
    print(f"Lowest Model: {lowest_model[1]} {lowest_model[0]['lowest_val_loss']}")
