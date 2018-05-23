'''
File that creates the neural networks
'''
from models.arima import arima_create
from models.lstm import lstm_create
from models.mlp import mlp_create
import multiprocessing
from functools import partial
import json

# configs = json.loads(open('config_lstm.json').read())
# def create_epoch():
#     return [1,20,40]
#
# def create_inputnodes():
#     return [50, 150]
#
# def create_lookback():
#     return [x for x in range(1,31,10)]
#
# def create_leadtime():
#     return [x for x in range(1,41,10)]
#
# def create_optimizers():
#     return ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
#
# def create_activationfunctions():
#     return ['softmax', 'elu', 'selu', 'softplux', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

epochs = 15
inputnodes = 50 # size of first
look_back = 1 # number of look_back units for LSTM network
time_steps = 3 # depends on the number of time_steps already used in "main.py". If you change it without considering what was used previously, results make no sense (e.g. predict values in the past...)
predict_var = 'target_t+0' # here the target column has not been shifted. It must be shifted with the same time_steps as in the previous steps of feature extraction.

# Must comment two of the following three:
# filename = 'datasets/8hmax-extracted_44201_0069-3_pca_2000-2016.csv'; normalize_X = False # PCA dataset
filename = 'datasets/8hmax-extracted_44201_0069-3_decTree_2000-2016.csv'; normalize_X = False # Decision Tree dataset
# filename = 'datasets/8hmax-prepared_44201_0069-3_2000-2016.csv'; normalize_X = True # Normal dataset with no extrac feature


lstm_create(20, inputnodes, look_back, predict_var, time_steps, filename=filename, normalize_X=normalize_X)
mlp_create(100, inputnodes, predict_var, time_steps, normalize_X=normalize_X, filename=filename)