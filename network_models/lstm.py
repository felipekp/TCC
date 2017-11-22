'''
LSTM RNN for predicting timeseries
Original code by Brian
Modified by Felipe Ukan

'''
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import math
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import utils.utils as utils

# # convert an array of values into a dataset matrix
# def create_3d_lookback_array(data, look_back):
#     '''
#         timestep/lookback is now: size given+1 because Im considering itself a different timestep (so, given 2 it will take the two past readings from the current and the current)
#         params:
#         :param data: actual array with all the data
#         :param look_back: int number that tells how many timesteps behind to look
#     '''
#     #determine number of data samples
#     rows_data,cols_data = np.shape(data)
    
#     #determine number of row to iterate
#     tot_batches = int(rows_data)
    
#     #initialize 3D tensor
#     threeD = np.zeros(((tot_batches-look_back,look_back+1,cols_data)))
    
#     # populate 3D tensor
#     for sample_num in range(look_back, tot_batches):
#         try:
#             threeD[sample_num-look_back,:,:] = data[(sample_num-look_back):sample_num+1,:]
#         except:
#             print 'ERROR: not able to add current element to the threeD array'

#     return threeD

# def remove_other_site_cols(df, site):
#     for col in df.columns:
#         # print col.split('_')[1]
#         if col.split('_')[1] != site:
#             del df[col]

def lstm_create(epochs, input_nodes, look_back, timesteps_ahead, predict_var, filename='datasets/kuwait.csv', optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
    """

    Given an csv file with all parameters and a 
    """
    # fix random seed for reproducibility
    np.random.seed(7)

    df = utils.read_csvdata(filename)

    # separates into axisY = X and axisY = Y
    axisX, axisY = utils.create_XY_arrays(df, look_back, timesteps_ahead, predict_var)

    # normalize the datasets
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))

    axisX = scalerX.fit_transform(axisX)
    axisY = scalerY.fit_transform(axisY)

    # prepare output arrays
    trainX, testX, trainY, testY = utils.prepare_XY_arrays(axisX, axisY, train_split, look_back)

    # ***
    # Network declaration
    # ***
    model = utils.createnet_lstm1(input_nodes, trainX)


    # compiles the model
    model.compile(loss=loss_function, optimizer=optimizer)

    # fits the model
    # history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), shuffle=False)

    #evaluates the model
    loss = model.evaluate(testX, testY)

    
    # test loss and training loss graph. It can help understand the optimal epochs size and if the model
    # is overfitting or underfitting.
    print ''
    print 'Loss (MSE):', loss
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

    # calculates MAE
    trainScore = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.5f MAE' % (trainScore))
    testScore = mean_absolute_error(testY, testPredict)
    print('Test Score: %.5f MAE' % (testScore))

    # plot baseline and predictions
    plt.close('all')
    plt.plot(testY, label='real_data')
    plt.plot(testPredict, label='prediction')
    plt.legend()
    # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
    plt.show()


# def lstm_create_old(epochs, input_nodes, look_back, timesteps_ahead, target_col_num=False, filename='datasets/kuwait.csv', optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
#     """

#     Given an csv file with all parameters and a 
#     """
#     # fix random seed for reproducibility
#     np.random.seed(7)

#     # reads input file an initializes it
#     df = utils.read_csvdata(filename)

#     # TODO: modify code to use only the default merged dataset.. and then here create the shifted column from the targetcolnum will become the number of the parameter..

#     # select target column
#     target_col = utils.select_column(len(list(df)), target_col_num)

#     # separates into axisY = X and axisY = Y
#     axisX, axisY = utils.create_XY_arrays(df, target_col) 

#     # normalize the datasets
#     scalerX = MinMaxScaler(feature_range=(0, 1))
#     scalerY = MinMaxScaler(feature_range=(0, 1))

#     axisX = scalerX.fit_transform(axisX)
#     axisY = scalerY.fit_transform(axisY)

#     # prepare output arrays
#     trainX, testX, trainY, testY = utils.prepare_XY_arrays(axisX, axisY, train_split, look_back)

#     # ***
#     # Network declaration
#     # ***
#     model = Sequential()

#     model.add(LSTM(input_nodes, return_sequences=True, activation='tanh', recurrent_activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))

#     model.add(Dropout(0.2))

#     model.add(LSTM(trainX.shape[2], activation='tanh', recurrent_activation='tanh', return_sequences=False))

#     # 1 neuron on the output layer
#     model.add(Dense(1, activation='linear'))

#     # compiles the model
#     model.compile(loss=loss_function, optimizer=optimizer)

#     # fits the model
#     # history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
#     history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), shuffle=False)

#     #evaluates the model
#     loss = model.evaluate(testX, testY)

    

#     # test loss and training loss graph. It can help understand the optimal epochs size and if the model
#     # is overfitting or underfitting.
#     print ''
#     print 'Loss (MSE):', loss
#     plt.plot(history.history['val_loss'], label='train')
#     plt.plot(history.history['loss'], label='validation')
#     plt.legend()
#     plt.show()

#     # make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)

#     # invert predictions
#     trainPredict = scalerY.inverse_transform(trainPredict)
#     trainY = scalerY.inverse_transform(trainY)
#     testPredict = scalerY.inverse_transform(testPredict)
#     testY = scalerY.inverse_transform(testY)

#     # calculates MAE
#     trainScore = mean_absolute_error(trainY, trainPredict)
#     print('Train Score: %.5f MAE' % (trainScore))
#     testScore = mean_absolute_error(testY, testPredict)
#     print('Test Score: %.5f MAE' % (testScore))

#     # calculate root mean squared error. 
#     # weights "larger" errors more by squaring the values when calculating
#     # print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
#     # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
#     # print('Train Score: %.5f RMSE' % (trainScore))
#     # testScore = math.sqrt(mean_squared_error(testY, testPredict))
#     # print('Test Score: %.5f RMSE' % (testScore))

#     # plot baseline and predictions
#     plt.close('all')
#     plt.plot(testY, label='real_data')
#     plt.plot(testPredict, label='prediction')
#     plt.legend()
#     # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
#     plt.show()
