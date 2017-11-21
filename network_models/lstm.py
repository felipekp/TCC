'''
LSTM RNN for predicting timeseries
Original code by Brian
Modified by Felipe Ukan

'''
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import utils.utils as utils

# convert an array of values into a dataset matrix
def create_3d_lookback_array(data, look_back):
    '''
        timestep/lookback is now: size given+1 because Im considering itself a different timestep (so, given 2 it will take the two past readings from the current and the current)
        params:
        :param data: actual array with all the data
        :param look_back: int number that tells how many timesteps behind to look
    '''
    #determine number of data samples
    rows_data,cols_data = np.shape(data)
    
    #determine number of row to iterate
    tot_batches = int(rows_data)
    
    #initialize 3D tensor
    threeD = np.zeros(((tot_batches-look_back,look_back+1,cols_data)))
    
    # populate 3D tensor
    for sample_num in range(look_back, tot_batches):
        try:
            threeD[sample_num-look_back,:,:] = data[(sample_num-look_back):sample_num+1,:]
        except:
            print 'ERROR: not able to add current element to the threeD array'

    return threeD

def remove_other_site_cols(df, site):
    for col in df.columns:
        # print col.split('_')[1]
        if col.split('_')[1] != site:
            del df[col]

def lstm_create(epochs, input_nodes, look_back, target_col_num=False, filename='datasets/kuwait.csv', optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
    # 8haverage-merged_2000-2016
    # fix random seed for reproducibility
    np.random.seed(7)
    # target_col_num = 6

    # reads input file an initializes it
    df = utils.read_csvdata(filename)

    # select target column
    target_col = utils.select_column(len(list(df)), target_col_num)

    # separates into axisY = X and axisY = Y
    axisX, axisY = utils.create_XY_arrays(df, target_col) 

    # normalize the datasets
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))

    axisX = scalerX.fit_transform(axisX)
    axisY = scalerY.fit_transform(axisY)

    # prepare output arrays
    trainX, testX, trainY, testY = utils.prepare_XY_arrays(axisX, axisY, train_split, look_back)

    print trainX

    exit()

    # ***
    # Network declaration
    # ***
    model = Sequential()

    model.add(LSTM(input_nodes, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))

    model.add(Dropout(0.2))

    model.add(LSTM(trainX.shape[2], activation='sigmoid', recurrent_activation='tanh', return_sequences=False))

    # 1 neuron on the output layer
    model.add(Dense(1, activation='sigmoid'))

    # compiles the model
    model.compile(loss=loss_function, optimizer=optimizer)

    # ***
    # 5) Increased the batch_size to 72. This improves training performance by more than 50 times
    # and loses no accuracy (batch_size does not modify the final result, only how memory is handled)
    # ***
    # history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), shuffle=False)

    loss = model.evaluate(testX, testY)

    print 'Loss (MSE):', loss

    # ***
    # 6) test loss and training loss graph. It can help understand the optimal epochs size and if the model
    # is overfitting or underfitting.
    # ***
    plt.plot(history.history['val_loss'], label='train')
    plt.plot(history.history['loss'], label='validation')
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

    # ***
    # 7) calculate mean absolute error. Different than root mean squared error this one
    # is not so "sensitive" to bigger erros (does not square) and tells "how big of an error"
    # we can expect from the forecast on average"
    # ***
    trainScore = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.5f MAE' % (trainScore))
    testScore = mean_absolute_error(testY[:len(testY)-20], testPredict[20:])
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
    plt.plot(testY, label='real_data')
    plt.plot(testPredict[20:], label='prediction')
    plt.legend()
    # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
    plt.show()
