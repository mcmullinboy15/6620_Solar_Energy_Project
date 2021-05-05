import random
import time
from datetime import datetime
import json
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn import preprocessing

from torch.optim import Adam, SGD, adadelta
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout

from Test import Test
from model_and_defs import Model1, Model2, Model3, Model4, calculate_Accuracy, plot, \
    getFileName, display_loss, separate_train_val_test


class Train:
    def __init__(self, data_file, epochs=20000, batch_size=10, Model=Model1,
                 H1=128, H2=None, H3=None, learning_rate=1e-4, Optim=torch.optim.Adam,
                 y_col=0, goal_loss=0.0000005, percent_disp=1.05, round_to=3, fast=True,
                 shuffle=False, index_col=None, test_size=50, save=True, DIR=None, NAME=None, scale=False,
                 headers=None, STD=False):
        self.fast = fast
        self.DIR = DIR
        self.NAME = NAME
        self.data_file = data_file
        self.scale = scale
        self.STD = STD

        self.preds = {'real': [], 'pred': []}
        self.loss = {'train': [], 'val': []}

        self.filename = f"{DIR}{NAME}_epoch_{epochs}_batch_{batch_size}_Model_{Model.__name__}_h1_{H1}_h2_{H2}_h3_{H3}_lr_{learning_rate}_optim_{Optim.__name__}"
        if save:
            print(save, 'save')
            if not os.path.exists(self.DIR):
                print('not exists')
                os.makedirs(self.DIR)

        self.START_TIME = datetime.now()

        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, index_col=index_col, header=headers)
            self.npa = np.array(self.df)
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
            self.npa = np.array(data_file)

        self.npa = self.npa.astype(dtype='float64')

        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.npa)

        self.data = separate_train_val_test(self.npa)
        self.info = f"Total: {len(self.npa)}, " \
                    f"Train: {len(self.data['train']['x'])}, " \
                    f"Validate: {len(self.data['val']['x'])}, " \
                    f"Test: {len(self.data['test']['x'])}\n"

        if not self.fast:
            print(self.info)
            print(f"Predicting Column :: {self.df.columns[y_col]} ::")

        self.train = self.data['train']
        self.x = self.train['x']
        self.y = self.train['y']
        self.y = self.y.reshape(-1, 1)

        self.val = self.data['val']
        self.xVal = self.val['x']
        self.yVal = self.val['y']
        self.yVal = self.yVal.reshape(-1, 1)

        if not self.fast:
            self.test = self.data['test']
            self.xtest = self.test['x']
            self.ytest = self.test['y']
            self.ytest = self.ytest.reshape(-1, 1)

        self._in = self.x.shape[1]
        self._out = self.y.shape[1]
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.BATCH_SIZE = batch_size
        self.GOAL_LOSS = goal_loss

        # this is just something I use in the accuracy stuff at the bottom,
        # if it's too low it'll cause problems
        self.ROUND_TO = round_to
        self.SAVE = save
        self.PERCENT_DISP = percent_disp

        # I don't print out the loss every time, I keep track of it in this
        # and I print it when it is the new lowest loss
        # but I also left it there if you wanted to un-comment it out
        self.loss_df = pd.DataFrame(columns=["loss"])
        self.lowest_val_loss = 10000000000

        self.MODEL = Model
        self.model = self.MODEL(self._in, self.H1, self._out, H2=self.H2, H3=self.H3)

        self.OPTIM = Optim
        self.criterion = nn.MSELoss()
        self.optimizer = self.OPTIM(self.model.parameters(), lr=learning_rate, weight_decay=0.5)

        if torch.cuda.is_available():
            self.model.cuda()

        # plt.ion()
        # plt.show()
        #
        # fig, (self.ax1, self.ax2) = plt.subplots(2, 1)

        print('Model Initialized')

    def core(self, phase, x, y):
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x).cuda()).float()
            labels = Variable(torch.from_numpy(y).cuda())
        else:
            inputs = Variable(torch.from_numpy(x)).float()
            labels = Variable(torch.from_numpy(y))

        self.optimizer.zero_grad()

        y_pred = self.model(inputs).double()

        loss_ = self.criterion(y_pred, labels)

        if phase == 'train':
            loss_.backward()
            self.optimizer.step()

        return loss_, y_pred

    def run(self, plot=False):
        # try:
        self.smallest_loss = 10000000.0
        self.ave_loss = 10000000.0

        loss_rows = {}

        train_loss_rows = {}
        train_loss = 1000000

        val_loss_rows = {}
        val_loss = 1000000

        # while ave_loss > GOAL_LOSS:
        for i in (range(self.epochs)):
            for phase in ['train', 'val']:
                if phase == 'train':
                    # print("=====  Training  =====")
                    # optimizer = scheduler(optimizer, epoch)
                    self.model.train(True)  # Set model to training mode
                else:
                    # print("===== Validating =====")
                    self.model.train(False)  # Set model to evaluate mode

                for j in range(0, len(self.data[phase]), self.BATCH_SIZE):
                    x = self.data[phase]['x']
                    y = self.data[phase]['y']

                    if phase == 'train':
                        x = x[j:j + self.BATCH_SIZE]
                        y = y[j:j + self.BATCH_SIZE]

                    if self.scale:
                        s = preprocessing.MinMaxScaler(feature_range=(0, 1))
                        row = np.append(y, x, 1)
                        shape = row.shape
                        row1 = s.fit_transform(row.reshape(-1, 1)).reshape(shape)

                        x = row1[:, 1:]
                        y = row1[:, 0].reshape(shape[0], 1)

                    if self.STD:
                        preprocessing.scale()

                    loss, pred = self.core(
                        phase=phase,
                        x=x,
                        y=y,
                    )

                    if phase is 'val':
                        #     print(f"LOSS: {loss}, PRED[0]: {pred[0].item()}, Y_ACTU[0]: {y[0].item()},")
                        # val_loss = loss.item()
                        # val_loss_rows.update(
                        #     {i: {'LOSS': val_loss, 'PRED[0]': pred[0].item(), 'Y_ACTU[0]': y[0].item()}})
                        # else:
                        #     print(f"LOSS: {loss}, PRED[0]: {pred[0].item()}, Y_ACTU[0]: {y[0].item()},")
                        # train_loss = loss.item()
                        # train_loss_rows.update(
                        #     {i: {'LOSS': train_loss, 'PRED[0]': pred[0].item(), 'Y_ACTU[0]': y[0].item()}})

                        if loss.item() < self.lowest_val_loss:
                            print(f"\n\nEpochs: {i}, Saving new Model: ", loss.item())
                            self.lowest_val_loss = loss.item()
                            self.finish()

                        # Std

                    self.loss[phase].append(loss.item())

                    self.preds['real'] = []
                    self.preds['pred'] = []

                    for y_, p_ in zip(y, pred):
                        self.preds['real'].append(y_.item())
                        self.preds['pred'].append(p_.item())

                    # loss_rows.update({i: {'val': val_loss, 'train': train_loss}})
            print(
                f"\r{i}/{self.epochs}, ave_loss: {self.ave_loss}, batch: {self.BATCH_SIZE}, Model: {self.MODEL.__name__}, h1: {self.H1}, h2: {self.H2}, h3: {self.H3}, lr: {self.learning_rate}, optim: {self.OPTIM.__name__}",
                end="")
            if plot:
                self.plot_loss(i)
                # self.plot_preds(i)
                # self.plot_loss_and_pred(i)

            if (float(sum(self.loss['val'])) / float(len(self.loss['val']))) < self.ave_loss:
                self.ave_loss = float(sum(self.loss['val'])) / float(len(self.loss['val']))
                if not self.fast:
                    print(self.ave_loss)
                # self.loss_df.append(loss_rows)
                # loss_rows = {}

        # except Exception as e:
        #     print(e)

        # self.finish()

    def finish(self):
        self.info += str(self)
        self.info += f"\nTime Info: \n" \
                     f"Start time: {self.START_TIME}, End time: {datetime.now()}, Length: {datetime.now() - self.START_TIME}\n\n"

        self.info += json.dumps({
            'data_file': self.data_file,
            'filename': self.filename,
            'MODEL': self.MODEL.__name__,
            '_in': self._in,
            '_out': self._out,
            'H1': self.H1,
            'H2': self.H2,
            'H3': self.H3,
            'optimizer': self.optimizer.__class__.__name__,
            'criterion': self.criterion.__class__.__name__,
            'ave_loss': self.ave_loss,
            'lowest_val_loss': self.lowest_val_loss,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'BATCH_SIZE': self.BATCH_SIZE,
        })

        self.save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'data_file': self.data_file,
            'filename': self.filename,
            'MODEL': self.MODEL,
            '_in': self._in,
            '_out': self._out,
            'H1': self.H1,
            'H2': self.H2,
            'H3': self.H3,
            'epochs': self.epochs,
            'BATCH_SIZE': self.BATCH_SIZE,
            'learning_rate': self.learning_rate,
            'ave_loss': self.ave_loss,
            'lowest_val_loss': self.lowest_val_loss,
        }

        if self.SAVE:
            self.save_model()

    def __str__(self):

        return f"" \
               f"\nData Used (df):\n" \
               f"{self.data_file}\n" \
               f"\nModel Used (model): \n" \
               f"{self.model}\n" \
               f"model saved to {self.filename}\n" \
               f"Smallest Loss {self.ave_loss}\n" \

    def save_model(self):
        print(f"{self.filename}-LOSS_{self.lowest_val_loss}.pth")
        torch.save(self.save_dict, f"{self.filename}.pth")
        # f = open(f'{self.filename}.txt', 'a+')
        # f.write(self.info)
        # f.close()

        if not self.fast:
            print(self.info)
            print()
            print(f'{self.filename}.pth')

    def plot_loss_and_pred(self, i):

        train_ = self.loss['train']
        val_ = self.loss['val']
        if i > 500:
            train_ = train_[500 + i]
            val_ = val_[500 + i]
        self.ax1.plot(train_, label='Train Loss')
        self.ax1.plot(val_, label='Val Loss')

        real = self.preds['real']  # if l < 100]
        pred = self.preds['pred']  # if l < 100]
        self.ax2.plot(real, label='Real Val')
        self.ax2.plot(pred, label='Pred Val')

        plt.legend()

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def plot_loss(self, i, diff=False):

        plt.ion()

        # diff = True
        size_plotted = 20
        train_ = self.loss['train']
        val_ = self.loss['val']

        if i > size_plotted:
            # just the last var(size_plotted)
            train_ = train_[-size_plotted:]
            val_ = val_[-size_plotted:]

            # get rid of the first  var(size_plotted)
            # train_ = train_[size_plotted:]
            # val_ = val_[size_plotted:]

        if diff:
            diffs = [va - tr for tr, va in zip(train_, val_)]
            plt.plot(diffs, label='Difference in losses')

        else:
            plt.plot(train_, label='Train Loss')
            plt.plot(val_, label='Val Loss')

        plt.legend()

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def plot_preds(self, i):
        plt.ion()
        # plt.show()

        real = self.preds['real']  # if l < 100]
        pred = self.preds['pred']  # if l < 100]

        plt.plot(real, label='Real Val')
        plt.plot(pred, label='Pred Val')

        plt.legend()

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

        # plt.ioff()
        # plt.show()

    def run_test(self):
        if self.SAVE:
            test_ = Test(self.filename)
        else:
            test_ = Test(loaded_dict=self.save_dict)
        try:
            test_._test()
            test_.plot(sym_0='', sym_1='')
            if self.SAVE:
                test_.save()
        except Exception as e:
            if not self.fast:
                print(f"\033[91m {e}\033[00m")

# if __name__ is '__main__':
# file = "data/Compiled_Data.csv"

# trains = [
#     (10000, 10),
#     # (10, 1000), (100, 1000), (1000, 1000), (10000, 1000)
# ]
# for epo, bat in trains:
#     train = Train(learning_rate=1e-4, data_file=file, Model=Model4, epochs=epo, batch_size=bat, H1=128, H2=256, save=False)
#     train.run()
#     train.run_test()
#     time.sleep(20)
