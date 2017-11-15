"""
    This script constains methods to 
        1. transform a series problem into a supervised learning problem.


    useful links:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare(p_start_year, p_end_year, county, prepare_input_path, prepare_output_path, state='48', site='0069'):
    # to select folder:
    global start_year, end_year
    start_year = p_start_year
    end_year = p_end_year
    root_dir = str(state + '/'+ county + '/' + merge_input_path + start_year + '-' + end_year + '/')

    # concatenates each file and resets the index to: 0, 1, 2, 3 ...
    df = pd.concat((pd.read_csv(os.path.join(root_dir, f)) for f in os.listdir(
        root_dir)), axis=1, join='outer').set_index('date').reset_index(drop=True)

    # pre-process data to use minMaxscaller
    # scaler = MinMaxScaler()
    # new_df = df
    # for column in df:
    #     new_df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))

    # print df
    df.to_csv(prepare_output_path + start_year + '-' + end_year + '.csv')
