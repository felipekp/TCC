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

# load dataset
start_year = '2000'
end_year = '2016'
site = '0069'

# data_file_name = 'merged/max-merged_' + start_year + '-' + end_year + '.csv'
data_file_name = 'out/pca.csv'
df = read_csv(data_file_name, engine='python', skipfooter=3)
df = df.set_index(df.columns[0])
df.index.rename('id', inplace=True)

# remove_other_site_cols(df, site)

a = list(df)

for i in range (len(a)):
    print i, a[i]

# pick column to predict
try:
    target_col = int(raw_input("Select the column number to predict (default = " + a[len(a)-1] + "): "))
except ValueError:
    target_col = len(a)-1   #choose last column as default

# choose look-ahead to predict   
try:
    lead_time =  int(raw_input("How many days ahead to predict (default = 2)?: "))
except ValueError:
    lead_time = 2

# pre process data
dataset1 = df.fillna(0).values
dataplot1 = dataset1[lead_time:, target_col]  # extracts the target_col
dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
# deletes target_column data
dataset1 = np.delete(dataset1, target_col, axis=1) # removes target_col from training dataset
dataset1 = dataset1.astype('float32')


scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

dataset = scalerX.fit_transform(dataset1)
dataplot = scalerY.fit_transform(dataplot1)
    
    
# split into train and test sets
splits = TimeSeriesSplit(n_splits=3)
plt.figure(1)
index = 1
for train_index, test_index in splits.split(dataplot):
    
    train, test = dataset[train_index], dataset[test_index]
    trainY, testY = dataplot[train_index], dataplot[test_index]

    # get number of epochs
    try:
        n_epochs = int(raw_input("Number of epochs? (Default = 10)? "))
    except ValueError:
        n_epochs = 10
    
    # prepare input Tensors
    try:
        look_back = int(raw_input("Number of recurrent (look-back) units? (Default = 1)? "))
    except ValueError:
        look_back = 1
    trainX = TensorForm(train, look_back)
    testX = TensorForm(test, look_back)

    input_nodes = 50

    # trim target arrays to match input lengths
    if len(trainX) < len(trainY):
        trainY = np.asmatrix(trainY[:len(trainX)])
        
    if len(testX) < len(testY):
        testY = np.asmatrix(testY[:len(testX)])

    model = Sequential()

    model.add(LSTM(input_nodes, activation='sigmoid', recurrent_activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))

    # 1 neuron on the output layer
    model.add(Dense(1))

    # compiles the model
    model.compile(loss='mean_squared_error', optimizer='nadam')

    history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=72, validation_data=(testX, testY), shuffle=False)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scalerY.inverse_transform(trainPredict)
    trainY = scalerY.inverse_transform(trainY)
    testPredict = scalerY.inverse_transform(testPredict)
    testY = scalerY.inverse_transform(testY)

    # calculates mean absolute error. 
    print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
    trainScore = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.5f MAE' % (trainScore))
    testScore = mean_absolute_error(testY, testPredict)
    print('Test Score: %.5f MAE' % (testScore))

    # calculates root mean squared error. 
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
