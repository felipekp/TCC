'''
LSTM RNN for predicting timeseries
Original code by Brian
Modified by Felipe Ukan

'''
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import utils.utils as utils


def lstm_create(epochs, input_nodes, look_back, predict_var, time_steps, filename, normalize_X, optimizer='nadam', testtrainlossgraph=False, batch_size=512, loss_function='mse', train_split=0.8):
    """

    Given an csv file with all parameters and a 
    """
    # fix random seed for reproducibility
    np.random.seed(7)

    # reads csv file and sets index column
    df = utils.read_csvdata(filename)

    col_to_drop = 'target_t+3'
    df.drop(col_to_drop, axis=1, inplace=True)

    # separates into axisX and axisY the input data
    axisX, axisY = utils.create_XY_arrays(df, look_back, predict_var, time_steps)

    # normalize the datasets
    if normalize_X:
        scalerX = MinMaxScaler(feature_range=(0, 1))
        axisX = scalerX.fit_transform(axisX)

    scalerY = MinMaxScaler(feature_range=(0, 1))
    axisY = scalerY.fit_transform(axisY)



    # prepare output arrays
    trainX, testX, trainY, testY = utils.prepare_XY_arrays(axisX, axisY, train_split, look_back)

    # Network declaration
    model = utils.createnet_lstm1(input_nodes, trainX)

    # compiles the model
    model.compile(loss=loss_function, optimizer=optimizer)

    # fits the model
    # history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), shuffle=False, verbose=0)

    #evaluates the model
    loss = model.evaluate(testX, testY)

    # test loss and training loss graph. It can help understand the optimal epochs size and if the model is overfitting or underfitting.
    utils.create_testtrainingloss_graph(history, loss)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scalerY.inverse_transform(trainPredict)
    trainY = scalerY.inverse_transform(trainY)
    testPredict = scalerY.inverse_transform(testPredict)
    testY = scalerY.inverse_transform(testY)

    print('Lookback:', look_back)

    # calculates MAE score
    utils.calculate_MAE(trainY, trainPredict, testY, testPredict)

    # calculates RMSE
    utils.calculate_RMSE(trainY, trainPredict, testY, testPredict)

    # creates graph with real test data and the predicted data
    utils.create_realpredict_graph(testY, testPredict)

