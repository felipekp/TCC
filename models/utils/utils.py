import numpy as np
import matplotlib.pyplot as plt

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def remove_other_site_cols(df, site):
    for col in df.columns:
        # print col.split('_')[1]
        if col.split('_')[1] != site:
            del df[col]

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


def create_XY_arrays(df, look_back, predict_var):
    """
        timesteps_ahead + look_back = actual timesteps the target should be moved from the first element
        int = look_back and timesteps_ahead
    """
    
    # creates new shifted column, select column 
    # df[predict_var + '_t+' + str(timesteps_ahead)] = df[predict_var].shift((timesteps_ahead + look_back)*-1)
    # df.dropna(inplace=True)

    # print(df.head(10))

    target_col = len(list(df))-1 # target column is the last one

    dataset1 = df.fillna(0).values
    dataplot1 = dataset1[:, target_col]  # extracts the target_col
    dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
    # deletes target_column data
    dataset1 = np.delete(dataset1, target_col, axis=1) # removes target_col from training dataset
    # dataset1.drop(target_col, axis=1, inplace=True)
    dataset1 = dataset1.astype('float32')

    return dataset1, dataplot1


def prepare_XY_arrays(axisX, axisY, train_split, look_back):
    train_size = int(len(axisX) * train_split)
    test_size = len(axisX) - train_size
    train, test = axisX[0:train_size], axisX[train_size:len(axisX)]

    # prepare output arrays
    trainY, testY = axisY[0:train_size], axisY[train_size:len(axisY)]

    n,p = np.shape(trainY)
    if n < p:
        trainY = trainY.T
        testY = testY.T

    # resize input sets
    trainX1 = train[:len(trainY),]
    testX1 = test[:len(testY),]
        
    # prepare input Tensors
    trainX = create_3d_lookback_array(trainX1, look_back)
    testX = create_3d_lookback_array(testX1, look_back)

    # trims target arrays to match input lengths
    if len(trainX) < len(trainY):
        trainY = np.asmatrix(trainY[:len(trainX)])
        
    if len(testX) < len(testY):
        testY = np.asmatrix(testY[:len(testX)])

    return trainX, testX, trainY, testY

def read_csvdata(filename):
    df = read_csv(filename, engine='python', skipfooter=3)
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    print 'dataset used:', filename

    return df

def createnet_lstm1(input_nodes, trainX):
    model = Sequential()

    model.add(LSTM(input_nodes, return_sequences=True, activation='tanh', recurrent_activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))

    model.add(Dropout(0.2))

    model.add(LSTM(trainX.shape[2], activation='tanh', recurrent_activation='tanh', return_sequences=False))

    # 1 neuron on the output layer
    model.add(Dense(1, activation='linear'))

    return model

def createnet_mlp1(input_nodes, trainX):
    model = Sequential()

    model.add(Dense(input_nodes, input_shape=(trainX.shape[1], trainX.shape[2]), activation='linear'))

    model.add(Dropout(0.2))

    model.add(Dense(20, activation='tanh'))

    model.add(Dropout(0.2))

    model.add(Dense(20, activation='tanh'))

    model.add(Dropout(0.2))

    model.add(Dense(50, activation='tanh'))

    model.add(Dropout(0.2))

    model.add(Dense(20, activation='tanh'))

    # model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1, activation='linear'))

    return model


def create_testtrainingloss_graph(history, loss):
    print ''
    print 'Loss (MSE):', loss
    plt.plot(history.history['val_loss'], label='train')
    plt.plot(history.history['loss'], label='validation')
    plt.legend()
    plt.show()


def create_realpredict_graph(testY, testPredict):
    # plot baseline and predictions
    plt.close('all')
    plt.plot(testY, label='real_data')
    plt.plot(testPredict, label='prediction')
    plt.legend()
    # plt.savefig('images_lstm_out/' + str(testScore) + '-' + str(epochs) + '-' + str(input_nodes) + '-' + str(look_back) + '-' + str(lead_time) + '-' + '_lstm.png')
    plt.show()

def calculate_MAE(trainY, trainPredict, testY, testPredict):
    # calculates MAE
    trainScore = mean_absolute_error(trainY, trainPredict)
    print('Train Score: %.5f MAE' % (trainScore))
    testScore = mean_absolute_error(testY, testPredict)
    print('Test Score: %.5f MAE' % (testScore))


def calculate_RMSE(trainY, trainPredict, testY, testPredict):
    # calculates RMSE
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.5f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.5f RMSE' % (testScore))