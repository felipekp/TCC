'''
Function to generate parameters and call LSTM model function
use lstm.py
'''
from models.arima import arima_create
from models.lstm import lstm_create
from models.mlp import mlp_create
import multiprocessing
from functools import partial
import json

def create_epoch():
    return [1,20,40]

def create_inputnodes():
    return [50, 150]

def create_lookback():
    return [x for x in range(1,31,10)]

def create_leadtime():
    return [x for x in range(1,41,10)]

def create_optimizers():
    return ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

def create_activationfunctions():
    return ['softmax', 'elu', 'selu', 'softplux', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

# target_col_num = 25 # manually set might try to make it the last by default, need the size first, so just inside the function
# # filename = 'datasets/kuwait.csv'
# filename = 'datasets/max-merged_2000-2016.csv'
# optimizer = 'nadam'
# testtrainlossgraph = True
# batch_size = 512
# loss_function = 'mse'
# train_split = 0.8 # test_splt is = 1 - train_split, always.

# pool = multiprocessing.Pool(4)

# for epochs in create_epoch():
#     for input_nodes in create_inputnodes():
#         for look_back in create_lookback():
#             # for lead_time in create_leadtime():
#                 # complexity x^4
#             print epochs, input_nodes, look_back
#             func = partial(lstm_create, epochs, input_nodes, look_back)  TODO: REMOVE LEADTIME
#             pool.map(func, create_leadtime())
#                 # lstm_create(epochs, input_nodes, look_back, lead_time, filename=filename)

# pool.close() 
# pool.join() 
configs = json.loads(open('config_lstm.json').read())
epochs = configs['model']['epochs']
inputnodes = configs['model']['input_nodes']
look_back = 5
# filename = 'datasets/8hmax-extracted_44201_0069-3_pca_2000-2016.csv'; normalize_X = False
filename = 'datasets/8hmax-extracted_44201_0069-3_decTree_2000-2016.csv'; normalize_X = False
# filename = 'datasets/8hmax-prepared_44201_0069-3_2000-2016.csv'; normalize_X = True
time_steps = 3
# filename = 'datasets/8haverage-merged_2000-2016.csv'
predict_var = 'target_t+0' #

# TODO:implement a report kind of printing system... so I know exactly with which parameters everything is being executed.

# epochs, inputnodes, lookback, leadtime

for item in range(5,30):
    look_back = item
    lstm_create(20, inputnodes, look_back, predict_var, time_steps, filename=filename, normalize_X=normalize_X)
# for optimizer in create_optimizers():
# mlp_create(100, inputnodes, filename=filename, optimizer='nadam')
# mlp_create(100, inputnodes, predict_var, time_steps, look_back=0, normalize_X=normalize_X, filename=filename)