"""
    This script constains methods to 
        1. transform a series problem into a supervised learning problem.


    useful links:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/
"""
from pandas import read_csv
import matplotlib as plt
import numpy as np
import pandas as pd
import utils.utils as utils

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     """
#     Frame a time series as a supervised learning dataset.
#     Arguments:
#         data: Sequence of observations as a list or NumPy array.
#         n_in: Number of lag observations as input (X).
#         n_out: Number of observations as output (y).
#         dropnan: Boolean whether or not to drop rows with NaN values.
#     Returns:
#         Pandas DataFrame of series framed for supervised learning.
#     """
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

@utils.timeit
def prepare(p_start_year, p_end_year, county, prepare_input_path, p_prepare_output_path, predict_var, p_timesteps='1', state='48', site='0069'):
    # to select file:
    start_year = p_start_year
    end_year = p_end_year
    filename = str(prepare_input_path + start_year + '-' + end_year + '.csv')
    timesteps = p_timesteps
    prepare_output_path = p_prepare_output_path + predict_var + '-' + timesteps + '_'
    

    # reading file
    df = read_csv(filename, engine='python', skipfooter=3)
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    # shifting the target parameter forward timesteps times
    df[predict_var + '_t+' + timesteps] = df[predict_var].shift(int(timesteps)*-1)
    df.dropna(inplace=True)

    # saves changes to new file
    df.to_csv(prepare_output_path + start_year + '-' + end_year + '.csv')
