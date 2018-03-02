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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import utils.utils as utils


def mlp_create(epochs, input_nodes, look_back, timesteps_ahead, predict_var, filename='datasets/kuwait.csv', optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
    # 8haverage-merged_2000-2016
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
    model = utils.createnet_mlp1(input_nodes, trainX)
    

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
    plt.plot(testPredict, label='prediction')
    plt.legend()
    # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
    plt.show()




# def mlp_create_old(epochs, input_nodes, target_col_num=False, filename='datasets/kuwait.csv', optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
#     # 8haverage-merged_2000-2016
#     # fix random seed for reproducibility
#     np.random.seed(7)
#     # target_col_num = 6

#     df = read_csv(filename, engine='python', skipfooter=3)
#     df = df.set_index(df.columns[0])
#     df.index.rename('id', inplace=True)

#     # try:
#     if type(target_col_num) == int: # verify if it is an integer
#         target_col = target_col_num
#     # except:
#     else:
#         target_col = len(list(df))-1   #choose last column as default

#     # ***
#     # 2) Creating and separating target dataset (as dataplot1) and training (as dataset1), pay attention that target_col must be removed from the training dataset!
#     # ***
#     dataset1 = df.fillna(0).values
#     dataplot1 = dataset1[0:, target_col]  # extracts the target_col
#     dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
#     # deletes target_column data
#     dataset1 = np.delete(dataset1, target_col, axis=1) # removes target_col from training dataset
#     dataset1 = dataset1.astype('float32')

#     # normalize the dataset
#     scalerX = MinMaxScaler(feature_range=(0, 1))
#     scalerY = MinMaxScaler(feature_range=(0, 1))

#     dataset = scalerX.fit_transform(dataset1)
#     dataplot = scalerY.fit_transform(dataplot1)
        

#     train_size = int(len(dataset) * train_split)
#     test_size = len(dataset) - train_size
#     train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

#     # prepare output arrays
#     trainY, testY = dataplot[0:train_size], dataplot[train_size:len(dataplot)]

#     n,p = np.shape(trainY)
#     if n < p:
#         trainY = trainY.T
#         testY = testY.T

#     # resize input sets
#     trainX = train[:len(trainY),]
#     testX = test[:len(testY),]
        
#     # prepare input Tensors
#     # trainX = TensorForm(trainX1, look_back)
#     # testX = TensorForm(testX1, look_back)
#     # # prepare final data for training
#     # # trainX = trainX1.reshape((trainX1.shape[0], 1, trainX1.shape[1]))
#     # # testX = testX1.reshape((testX1.shape[0], 1, testX1.shape[1]))

#     # # trim target arrays to match input lengths
#     # if len(trainX) < len(trainY):
#     #     trainY = np.asmatrix(trainY[:len(trainX)])
        
#     # if len(testX) < len(testY):
#     #     testY = np.asmatrix(testY[:len(testX)])


#     # print(trainX.shape, trainY.shape, testX.shape, testY.shape)

#     # print trainX.shape

#     # exit()

#     model = utils.createnet_mlp1()

#     # compiles the model
#     model.compile(loss=loss_function, optimizer=optimizer)

#     # ***
#     # 5) Increased the batch_size to 72. This improves training performance by more than 50 times
#     # and loses no accuracy (batch_size does not modify the final result, only how memory is handled)
#     # ***
#     history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), shuffle=False)

#     loss = model.evaluate(testX, testY)

#     print 'Loss (MSE):', loss

#     # ***
#     # 6) test loss and training loss graph. It can help understand the optimal epochs size and if the model
#     # is overfitting or underfitting.
#     # ***
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

#     # ***
#     # 7) calculate mean absolute error. Different than root mean squared error this one
#     # is not so "sensitive" to bigger erros (does not square) and tells "how big of an error"
#     # we can expect from the forecast on average"
#     # ***
#     trainScore = mean_absolute_error(trainY, trainPredict)
#     print('Train Score: %.5f MAE' % (trainScore))
#     testScore = mean_absolute_error(testY[:len(testY)-3], testPredict[3:])
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
#     plt.plot(testPredict[3:], label='prediction')
#     plt.legend()
#     # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
#     plt.show()
