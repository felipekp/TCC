'''
ARIMA for predicting timeseries
Original code: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
Modified by Felipe Ukan

ARIMA:
AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

PARAMETERS:
p: The number of lag observations included in the model, also called the lag order.
d: The number of times that the raw observations are differenced, also called the degree of differencing.
q: The size of the moving average window, also called the order of moving average.
'''

import utils.utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def arima_create(look_back, predict_var, filename='datasets/kuwait.csv'):

    df = utils.read_csvdata(filename)
    df = df[predict_var].dropna()
    df.drop(df.tail(2).index, inplace=True)

    print(df.head(15))

    dates = pd.to_datetime(
        pd.date_range('2000-01-03', '2016-12-31', freq='8H'))
    df.index = dates
    # df.index.rename('date', inplace=True)

    # print(df.head())

    size = int(len(df) * 0.66)
    train, test = df[0:size], df[size:len(df)]
    history = [x for x in train]
    predictions = list()

    #
    # for t in range(len(test)):
    #     model = ARIMA(history, order=(3, 1, 0))
    #     model_fit = model.fit(disp=0)
    #     output = model_fit.forecast()
    #     yhat = output[0]
    #     predictions.append(yhat)
    #     obs = test[t]
    #     history.append(obs)
    #     print('predicted=%f, expected=%f' % (yhat, obs))
    #
    # error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()

    # first example
    # model = ARIMA(df, order=(5,1,0))
    # model_fit = model.fit(disp=0)
    # print(model_fit.summary())
    #
    # residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()
    # plt.show()
    # residuals.plot(kind='kde')
    # plt.show()
    # print(residuals.describe())

    # autocorrelation plot
    print(df.head())
    # pd.plotting.autocorrelation_plot(df)
    # plt.show()
