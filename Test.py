import sys

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint

from sklearn.metrics import r2_score
from model_and_defs import Model1, Model2, Model3
from sklearn.preprocessing import MinMaxScaler


class Test:
    def __init__(self, path=None, data=None, fast=True, loaded_dict=None):
        self.fast = fast
        print("Testing")

        if path is not None:
            self.model_path = f"{path}.pth"
            self.loaded_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        else:
            self.loaded_dict = loaded_dict

        try:
            self.model_state_dict = self.loaded_dict['model_state_dict']
            self.optimizer_state_dict = self.loaded_dict['optimizer_state_dict']
            self.criterion_used = self.loaded_dict['criterion_state_dict']
            self.data_file = self.loaded_dict['data_file']
            self.epochs = self.loaded_dict['epochs']
            self.H1 = self.loaded_dict['H1']
            self.H2 = None
            self.H3 = None
            self._in = self.loaded_dict['_in']
            self._out = self.loaded_dict['_out']
            self.learning_rate = self.loaded_dict['learning_rate']
            self.filename = self.loaded_dict['filename']

            try:
                self.BATCH_SIZE = self.loaded_dict['BATCH_SIZE']
                self.H2 = self.loaded_dict['H2']
                self.H3 = self.loaded_dict['H3']
                self.ave_loss = self.loaded_dict['ave_loss']

            except:
                pass

            try:
                self.MODEL = self.loaded_dict['MODEL']
            except:
                pass

                self.Model_name = self.loaded_dict['Model']
                if self.Model_name.__contains__('1'):
                    self.Model = Model1
                elif self.Model_name.__contains__('2'):
                    self.Model = Model2
                elif self.Model_name.__contains__('3'):
                    self.Model = Model3
                else:
                    raise Exception(f"{self.Model_name} is not a real Model Name")

        except Exception as e:
            print(e)
            print(
                f"Error:  This model file({self.filename[self.filename.rindex('/'):]}) doesn't support the new Model Dict")

        if data is None:
            self.data = self.data_file
        else:
            self.data = data
        if isinstance(self.data, str):
            self.npa = np.array(pd.read_csv(self.data, index_col=None))
        elif isinstance(self.data, pd.DataFrame):
            self.npa = np.array(self.data)
        self.npa = self.npa.astype(dtype='float64')

        self.npa = self.separate_train_val_test(self.npa)

        self.test = self.npa['val']
        self.xtest = self.test['x']
        self.ytest = self.test['y']
        self.ytest = self.ytest.reshape(-1, 1)

        self.model = self.MODEL(self._in, self.H1, self._out, H2=self.H2, H3=self.H3)
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()

    def separate_y_x(self, data, y_col=0):
        x = data[:, y_col + 1:]
        y = data[:, y_col]
        y = y.reshape(-1, 1)

        return {'y': y, 'x': x}

    def separate_train_val_test(self, data, y_col=0, train_size=0.80, val_size=0.1):
        data = np.array(data)
        val_s = train_size + val_size
        train, validate, test = np.split(data, [int(train_size * len(data)), int((val_s) * len(data))])

        return {
            'train': self.separate_y_x(train, y_col),
            'val': self.separate_y_x(validate, y_col),
            'test': self.separate_y_x(test, y_col)
        }

    def _test(self):
        print("Running Test")

        with torch.no_grad():
            self.predicted = self.model(torch.from_numpy(self.xtest).float())  # .data.numpy()

            self.predicted = self.predicted.numpy()
            self.predicted = np.around(self.predicted, decimals=0)
            self.predicted[self.predicted < 1] = 1
            self.predicted[self.predicted > 5] = 5

            self.info = f"R2: {r2_score(self.ytest, self.predicted)}"
            if not self.fast:
                print(self.info, f"Lowest MSE: {self.ave_loss}")
                temp_dict = self.loaded_dict
                temp_dict.pop('model_state_dict')
                temp_dict.pop('optimizer_state_dict')
                pprint.PrettyPrinter(indent=4).pprint(temp_dict)

    def plot(self, label_0="True data", sym_0='go', label_1="Predictions", sym_1='x'):

        fig, axarr = plt.subplots(1, 1)
        axarr.plot(self.ytest, sym_0, label=label_0, alpha=0.5)
        axarr.plot(self.predicted, sym_1, label=label_1, alpha=0.5)
        axarr.legend(loc='best')

        self.fig_line = fig

        fig, axarr = plt.subplots(1, 1)
        axarr.plot(self.ytest, self.predicted, 'go', label=label_0, alpha=0.5)
        # axarr.plot( 'go', label=label_1, alpha=0.5)
        axarr.legend(loc='best')

        self.fig_dots = fig

        # plt.show()

    def accu(self):
        pass

    def save(self):

        # Saving Info
        print('\n\nFrom Test:\n', self.info, file=open(f"{self.filename}.txt", 'a+'))

        # Saving Plot
        self.fig_line.savefig(f"{self.filename}.png")
        # self.fig_dots.savefig(f"{self.filename}-dots.png")
        plt.close('all')
