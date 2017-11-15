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
from keras.layers import Dense, LSTM, Dropout
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

def lstm_create(epochs, input_nodes, look_back, lead_time, target_col_num=False, filename='datasets/kuwait.csv', optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
    # 8haverage-merged_2000-2016
    # fix random seed for reproducibility
    np.random.seed(7)
    # target_col_num = 6

    df = read_csv(filename, engine='python', skipfooter=3)
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    # try:
    if type(target_col_num) == int: # verify if it is an integer
        target_col = target_col_num
    # except:
    else:
        target_col = len(list(df))-1   #choose last column as default

    # ***
    # 2) Creating and separating target dataset (as dataplot1) and training (as dataset1), pay attention that target_col must be removed from the training dataset!
    # ***
    dataset1 = df.fillna(0).values
    dataplot1 = dataset1[lead_time:, target_col]  # extracts the target_col
    dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
    # deletes target_column data
    dataset1 = np.delete(dataset1, target_col, axis=1) # removes target_col from training dataset
    dataset1 = dataset1.astype('float32')

    # normalize the dataset
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))

    dataset = scalerX.fit_transform(dataset1)
    dataplot = scalerY.fit_transform(dataplot1)
        

    train_size = int(len(dataset) * train_split)
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
        
    # prepare input Tensors
    trainX = TensorForm(trainX1, look_back)
    testX = TensorForm(testX1, look_back)

    # trim target arrays to match input lengths
    if len(trainX) < len(trainY):
        trainY = np.asmatrix(trainY[:len(trainX)])
        
    if len(testX) < len(testY):
        testY = np.asmatrix(testY[:len(testX)])

    model = Sequential()

    # ***
    # 3) Actual change on the LSTM layer
    # ***
    model.add(LSTM(input_nodes, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))

    model.add(Dropout(0.2))

    model.add(LSTM(trainX.shape[2], activation='sigmoid', recurrent_activation='tanh', return_sequences=False))

    model.add(Dropout(0.2))

    # 1 neuron on the output layer
    model.add(Dense(1, activation='sigmoid'))

    # compiles the model
    model.compile(loss=loss_function, optimizer=optimizer)

    # ***
    # 5) Increased the batch_size to 72. This improves training performance by more than 50 times
    # and loses no accuracy (batch_size does not modify the final result, only how memory is handled)
    # ***
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)

    loss = model.evaluate(testX, testY)

    print 'Loss (MSE):', loss

    # ***
    # 6) test loss and training loss graph. It can help understand the optimal epochs size and if the model
    # is overfitting or underfitting.
    # ***
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scalerY.inverse_transform(trainPredict)
    trainY = scalerY.inverse_transform(trainY)
    testPredict = scalerY.inverse_transform(testPredict)
    testY = scalerY.inverse_transform(testY)

    # ***
    # 7) calculate mean absolute error. Different than root mean squared error this one
    # is not so "sensitive" to bigger erros (does not square) and tells "how big of an error"
    # we can expect from the forecast on average"
    # ***
    print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
    trainScore = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.5f MAE' % (trainScore))
    testScore = mean_absolute_error(testY, testPredict)
    print('Test Score: %.5f MAE' % (testScore))

    # calculate root mean squared error. 
    # weights "larger" errors more by squaring the values when calculating
    # print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
    # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    # print('Train Score: %.5f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY, testPredict))
    # print('Test Score: %.5f RMSE' % (testScore))

    # plot baseline and predictions
    plt.close('all')
    plt.plot(testY)
    plt.plot(testPredict)
    # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
    plt.show()
