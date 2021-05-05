import os
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np


class Model1(nn.Module):
    def __init__(self, D_in, H, D_out, H2=None, *args, **kwargs):
        super().__init__()

        if H2 is None:
            H2 = H

        self.l1 = nn.Linear(D_in, H)
        self.relu1 = nn.Linear(H, H2)
        self.relu2 = nn.Linear(H2, H2)
        self.relu3 = nn.Linear(H2, H)
        self.l2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.relu1(x))
        x = F.relu(self.relu2(x))
        x = F.relu(self.relu3(x))
        x = self.l2(x)
        return x


class Model2(nn.Module):
    def __init__(self, D_in, H, D_out, H2=None, H3=None):
        super().__init__()

        if H2 is None:
            H2 = H
        if H3 is None:
            H3 = H

        self.l1 = nn.Linear(D_in, H)
        self.relu1 = nn.Linear(H, H2)
        self.relu2 = nn.Linear(H2, H3)
        self.relu3 = nn.Linear(H3, H2)
        self.relu4 = nn.Linear(H2, H)
        self.l2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.relu1(x))
        x = F.relu(self.relu2(x))
        x = F.relu(self.relu3(x))
        x = F.relu(self.relu4(x))
        x = self.l2(x)
        return x


class Model3(nn.Module):
    def __init__(self, D_in, H, D_out, H2=None, *args, **kwargs):
        super().__init__()

        if H2 is None:
            H2 = H

        self.l1 = nn.Linear(D_in, H)
        self.relu1 = nn.Linear(H, H2)
        self.relu2 = nn.Linear(H2, H2)
        self.relu3 = nn.Linear(H2, H)
        self.dropout1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.relu1(x))
        x = F.relu(self.relu2(x))
        x = F.relu(self.relu3(x))
        x = self.dropout1(x)
        x = self.l2(x)
        return x


class Model4(nn.Module):
    def __init__(self, _in, H, _out, H2=None, H3=None):
        super(Model4, self).__init__()

        # converting 16 into 4
        _in = int(_in / 4)


        self.cnn_layer1 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(in_channels=_in, out_channels=H, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(H),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.cnn_layer2 = nn.Sequential(
            # Defining another 2D convolution layer
            nn.Conv1d(in_channels=H, out_channels=H, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(H),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(H * 3, _out)
        )

    # Defining the forward pass
    def forward(self, x):
        x = x.reshape(-1, 4, 4)
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def calculate_Accuracy(predicted, real, ROUND_TO, SAVE, f=None):
    correct, total = 0, 0
    fail_1, fail_2, fail_3, fail_4 = 0, 0, 0, 0

    # The print statements below will show which values failed.
    # It was here that I saw comparisions failed even when they
    # were only off by .001, But I understand that we're not going
    # for Accuracy. This all just helps see how off the calculations are.
    for idx, i in enumerate(predicted):
        i = round(i.data.numpy().item(), ROUND_TO)
        y = round(real[idx].item(), ROUND_TO)
        # print(f"{i == y}, {i} == {y}")

        if i == y:
            correct += 1
        elif round(i, ROUND_TO - 1) == round(y, ROUND_TO - 1):
            # print(f"FAIL[{ROUND_TO}] :: {i == y}, {i} == {y}")
            fail_1 += 1
            fail_2 += 1
            fail_3 += 1
            fail_4 += 1
        elif round(i, ROUND_TO - 2) == round(y, ROUND_TO - 2):
            # print(f"FAIL[{ROUND_TO - 1}] :: {i == y}, {i} == {y}")
            fail_2 += 1
            fail_3 += 1
            fail_4 += 1
        elif round(i, ROUND_TO - 3) == round(y, ROUND_TO - 3):
            # print(f"FAIL[{ROUND_TO - 2}] :: {i == y}, {i} == {y}")
            fail_3 += 1
            fail_4 += 1
        else:
            # print(f"FAIL[{ROUND_TO - 3}] :: {i == y}, {i} == {y}")
            fail_4 += 1

        total += 1

    acc_0 = round(correct / total, 3) * 100
    acc_1 = round((correct + fail_1) / total, 3) * 100
    acc_2 = round((correct + fail_2) / total, 3) * 100
    acc_3 = round((correct + fail_3) / total, 3) * 100

    accs = "Accuracies:\n" \
           f"R2 (R Squared) is [r, p]: {r2_score(real, predicted)}\n" \
           f"R2 (R Squared) is [p, r]: {r2_score(predicted, real)}\n" \
           f"Accuracy at {ROUND_TO} dec: {acc_0}%, " \
           f"total: {total}, Correct: {correct}\n" \
           f"Accuracy at {ROUND_TO - 1} dec: {acc_1}%, " \
           f"total: {total}, Correct: {(correct + fail_1)}\n" \
           f"Accuracy at {ROUND_TO - 2} dec: {acc_2}%, " \
           f"total: {total}, Correct: {(correct + fail_2)}\n" \
           f"Accuracy at {ROUND_TO - 3} dec: {acc_3}%, " \
           f"total: {total}, Correct: {(correct + fail_3)}\n"

    if SAVE:
        print(accs, file=f)
    print(accs)

    return max([acc_0, acc_1, acc_2, acc_3])


def plot(predicted, real, label_0="True data", sym_0='go', label_1="Predictions", sym_1='x', SAVE_IMG=True, f=None, filename=None):
    fig, axarr = plt.subplots(1, 1)
    axarr.plot(real, sym_0, label=label_0, alpha=0.5)  # ,
    axarr.plot(predicted, sym_1, label=label_1, alpha=0.5)
    axarr.legend(loc='best')

    if SAVE_IMG:
        if filename is not None:
            if f is not None:
                print(filename, file=f)
            fig.savefig(filename)

    plt.show()


def create_acc_loss_graph(model_name):
    contents = open('model.log', 'r').read().split('\n')

    times = []
    accuracies = []
    losses = []

    epochs = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, epoch, val_acc, val_loss = c.split(',')


def getFileName(ext, H, learning_rate, epochs, BATCH_SIZE, smallest_loss, folder=None, ):
    if folder is None:
        return f"{os.getcwd()}\\Outputs\\{H}_{learning_rate}_{epochs}_{smallest_loss}_{BATCH_SIZE}___.{ext}"
    else:
        return f"{os.getcwd()}\\Outputs\\{folder}\\{H}_{learning_rate}_{epochs}_{smallest_loss}_{BATCH_SIZE}___.{ext}"


def display_loss(ave_loss, loss, smallest_loss, START_TIME, PERCENT_DISP):
    # only prints the loss and the time info when we have a smaller loss
    if float(loss.item()) < smallest_loss:
        smallest_loss = float(loss.item())
        print(f"\nSMALLEST: {smallest_loss}, AVERAGE: {ave_loss}")
        print(f"Start time: {START_TIME}, Running for: {datetime.now() - START_TIME}")
    elif loss.item() < smallest_loss * PERCENT_DISP:
        print(loss.item())


def separate_y_x(data, y_col=0):
    x = data[:, y_col + 1:]
    y = data[:, y_col]
    y = y.reshape(-1, 1)

    return {'y': y, 'x': x}


def separate_train_val_test(data, y_col=0, train_size=0.80, val_size=0.1):
    data = np.array(data)
    val_s = train_size + val_size
    train, validate, test = np.split(data, [int(train_size * len(data)), int((val_s) * len(data))])

    dict = {
        'train': separate_y_x(train, y_col),
        'val': separate_y_x(validate, y_col),
        'test': separate_y_x(test, y_col)
    }
    return dict
