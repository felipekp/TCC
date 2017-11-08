'''
LSTM RNN for predicting timeseries
Original code by Brian
Modified by Felipe Ukan

'''
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

import pandas as pd
import math
from pandas import ExcelWriter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import json

# convert an array of values into a dataset matrix
def TensorForm(data, look_back):
    #determine number of data samples
    rows_data,cols_data = np.shape(data)
    
    #determine # of batches based on look-back size
    tot_batches = int(rows_data-look_back)+1
    
    #initialize 3D tensor
    threeD = np.zeros(((tot_batches,look_back,cols_data)))
    
    # populate 3D tensor
    for sample_num in range(tot_batches):
        for look_num in range(look_back):
            threeD[sample_num,:,:] = data[sample_num:sample_num+(look_back),:]
    
    return threeD

def remove_other_site_cols(df, site):
    for col in df.columns:
        # print col.split('_')[1]
        if col.split('_')[1] != site:
            del df[col]

# fix random seed for reproducibility
np.random.seed(7)

# configuration file
configs = json.loads(open('config_lstm.json').read())

# load dataset
df = read_csv(configs['data']['filename'], engine='python', skipfooter=3)
df = df.set_index(df.columns[0])
df.index.rename('id', inplace=True)

# target column  configuration
target_col = configs['data']['target_col']

# lead time configuration
lead_time = configs['data']['lead_time']

# ***
# 2) Creating and separating target dataset (as dataplot1) and training (as dataset1), pay attention that target_col must be removed from the training dataset!
# ***
dataset1 = df.fillna(0).values
dataplot1 = dataset1[lead_time:, target_col]  # extracts the target_col
dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
# deletes target_column data
dataset1 = np.delete(dataset1, target_col, axis=1) # removes target_col from training dataset
dataset1 = dataset1.astype('float32')

# normalize data
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

dataset = scalerX.fit_transform(dataset1)
dataplot = scalerY.fit_transform(dataplot1)
    
# slipt trans and test dataset
train_size = int(len(dataset) * configs['data']['train_test_split'])
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

# prepare output arrays
trainY, testY = dataplot[0:train_size], dataplot[train_size:len(dataplot)]

n,p = np.shape(trainY)
if n < p:
    trainY = trainY.T
    testY = testY.T

# resize input sets
trainX1 = train[:len(trainY),]
testX1 = test[:len(testY),]
  
# number of epochs
n_epochs = configs['model']['epochs']

# look back number
look_back = n_epochs = configs['model']['epochs'] 

# prepare input Tensors
trainX = TensorForm(trainX1, look_back)
testX = TensorForm(testX1, look_back)

# ***
# 4) number of neuros / input_nodes increased for the LSTM layer
# ***
input_nodes = configs['model']['input_nodes'] 

# trim target arrays to match input lengths
if len(trainX) < len(trainY):
    trainY = np.asmatrix(trainY[:len(trainX)])
    
if len(testX) < len(testY):
    testY = np.asmatrix(testY[:len(testX)])

# creating model
model = Sequential()

model.add(LSTM(input_nodes, activation='sigmoid', recurrent_activation='tanh', 
                input_shape=(trainX.shape[1], trainX.shape[2])))

# 1 neuron on the output layer
model.add(Dense(1))

# compiles the model
model.compile(loss=configs['model']['loss_function'], optimizer=configs['model']['optimiser_function'])


history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=configs['model']['batch_size'], 
    validation_data=(testX, testY), shuffle=False)

# test loss and training loss graph. It can help understand the optimal epochs size and if the model is overfitting or underfitting.
if configs['model']['testtraining_loss_graph']:
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform(trainY)
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform(testY)

# calculate mean absolute error. Different than root mean squared error this one is not so "sensitive" to bigger erros (does not square) and tells "how big of an error" we can expect from the forecast on average"
print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
trainScore = mean_absolute_error(trainY, trainPredict)
print('Train Score: %.5f MAE' % (trainScore))
testScore = mean_absolute_error(testY, testPredict)
print('Test Score: %.5f MAE' % (testScore))

# calculate root mean squared error. 
# weights "larger" errors more by squaring the values when calculating
print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.5f RMSE' % (testScore))

# plot baseline and predictions
plt.close('all')
plt.plot(testY)
plt.plot(testPredict)
plt.show()
