LSTM RNN for predicting timeseries


'''
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from pandas import ExcelWriter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
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

	
# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
data_file_name = 'brian_dataset.csv'
df = read_csv(data_file_name, engine='python', skipfooter=3)

last_col = np.shape(df)[1] - 1

target_col = last_col # last column is the target column

lead_time = 24
    
#convert to floating numpy arrays
dataset1 = df.fillna(0).values
dataset1 = dataset1.astype('float32')
dataplot1 = dataset1[lead_time:,target_col]  #shift training data
dataplot1 = dataplot1.reshape(-1,1)

# normalize the dataset
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

dataset = scalerX.fit_transform(dataset1)
dataplot = scalerY.fit_transform(dataplot1)
    
# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# prepare output arrays
trainY, testY = dataplot[0:train_size], dataplot[train_size:len(dataplot)]

n,p = np.shape(trainY)
if n < p:
    trainY = trainY.T
    testY = testY.T

# resize input sets
trainX1 = train[:len(trainY),]
testX1 = test[:len(testY),]
  
# number of epochs
n_epochs = 10
    
# prepare input Tensors
look_back = 8

# number of neuros / input_nodes
input_node = 50

trainX = TensorForm(trainX1, look_back)
testX = TensorForm(testX1, look_back)

# trim target arrays to match input lengths
if len(trainX) < len(trainY):
    trainY = np.asmatrix(trainY[:len(trainX)])
    
if len(testX) < len(testY):
    testY = np.asmatrix(testY[:len(testX)])

# start creating the model
model = Sequential()

# has input_nodes which is = to neurons, size here is = 26 by default
model.add(LSTM(input_node, activation='sigmoid', recurrent_activation='tanh', 
                input_shape=(trainX.shape[1], trainX.shape[2])))

# 1 neuron on the output layer
model.add(Dense(1))

# compiles the model
model.compile(loss='mean_squared_error', optimizer='nadam')

# trains the model
history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=72, validation_data=(testX, testY), shuffle=False)

# test loss and training loss graph
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
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

# calculate mean absolute error
print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
trainScore = mean_absolute_error(trainY, trainPredict)
print('Train Score: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(testY, testPredict)
print('Test Score: %.2f MAE' % (testScore))

# calculate root mean squared error
print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


# make timestamp for unique filname
stamp = str(time.clock())  #add timestamp for unique name
stamp = stamp[0:2] 

# generate filename and remove extra periods
filename = 'NEWFinErr_lstm_'+ str(n_epochs) + str(lead_time) + '_' + stamp + '.csv'    #example output file
if filename.count('.') == 2:
    filename = filename.replace(".", "",1)

# START: write results to csv file (TODO: verify if itś writing exactly what it was writing)
df_trainPredict = pd.DataFrame(trainPredict, columns=['trainPredict']) #save prediction output
df_obsTrain = pd.DataFrame(trainY, columns=['obsTrain']) #save observed output
df_testPredict = pd.DataFrame(testPredict, columns=['testPredict']) #save output training data
df_obsTest = pd.DataFrame(testY, columns=['obsTest'])
pd.concat([df_trainPredict, df_obsTrain, df_testPredict, df_obsTest], axis=1).to_csv(filename, index=False)
print'File saved in ', filename
# END: write results to csv file

# plot baseline and predictions
plt.close('all')
plt.plot(testY)
plt.plot(testPredict)
plt.show()
