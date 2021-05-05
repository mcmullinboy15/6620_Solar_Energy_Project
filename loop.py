import os

from tqdm.auto import tqdm
from Train import Train
from model_and_defs import Model1, Model2

from pprint import pprint


def run(trained, data, epoch, batch, Model, h1, h2, h3, lr, optim, NAME, skip_done, plot_train, DIR=None, STD=False):
    if DIR is None:
        DIR = f'{NAME}/{epoch}/{batch}/{Model.__name__}_{optim.__name__}/{h1}/{h2}/{h3}/{lr}/'
    else:
        DIR = f"{DIR}{batch}/"
    print('\n=====================================New Model============================================')
    print(
        f'epoch: {epoch}, batch: {batch}, Model: {Model.__name__}, h1: {h1}, h2: {h2}, h3: {h3}, lr: {lr}, optim: {optim.__name__}')
    print('==========================================================================================')
    """
         data_file, epochs=20000, batch_size=10, Model=Model1, H1=128, H2=None, H3=None, learning_rate=1e-4, optim=torch.optim.Adam
    """
    train = Train(data, epoch, batch, Model, h1, h2, h3, lr, optim,
                  # DIR=f'Outputs/{NAME}/{epoch}/{batch}/{Model.__name__}_{optim.__name__}/{h1}/{h2}/{h3}/{lr}/',
                  DIR=DIR,
                  NAME=NAME, STD=STD)
    print(f'Training : {train.filename}')
    if not os.path.exists(f'{train.filename}.pth') or not skip_done:# or epoch > 9000:
        print("running")
        train.run(plot_train)
        train.run_test()
    trained.append(train)

    print('\n====================================Model Complete===========================================')
    print(
        f'epoch: {epoch}, batch: {batch}, Model: {Model.__name__}, h1: {h1}, h2: {h2}, h3: {h3}, lr: {lr}, optim: {optim.__name__}')
    print('=============================================================================================')


def loop_train(DIR, NAME, paramaters, data="combined_data.csv", std=False, skip_done=True, plot_train=False):
    pprint(paramaters)
    # print(sum([len(params[i]) for i in params]))

    trained = []

    for epoch in tqdm(paramaters['EPOCHS']):
        for batch in tqdm(paramaters['BATCHES']):
            for Model in tqdm(paramaters['MODELS']):
                for h1 in paramaters['H1S']:
                    for h2 in paramaters['H2S']:
                        for lr in paramaters['LRS']:
                            for optim in paramaters['OPTIMS']:
                                if Model is Model2:
                                    for h3 in paramaters['H3S']:
                                        run(trained, data, epoch, batch, Model, h1, h2, h3, lr, optim, NAME, skip_done, plot_train=plot_train, STD=std, DIR=DIR)
                                else:
                                    run(trained, data, epoch, batch, Model, h1, h2, h3=128, lr=lr, optim=optim, STD=std, DIR=DIR, NAME=NAME, skip_done=skip_done, plot_train=plot_train)
    return trained
