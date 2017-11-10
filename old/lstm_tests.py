import time
import time
import threading
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
import json
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from pandas import read_csv
configs = json.loads(open('configs.json').read())

def plot_results(predicted_data, true_data):
    fig=plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig=plt.figure(figsize=(18, 12), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    
true_values = []
def generator_strip_xy(data_gen, true_values):
    for x, y in data_gen_test:
        true_values += list(y)
        yield x
    
def fit_model_threaded(model, data_gen_train, steps_per_epoch, configs):
    """thread worker for model fitting - so it doesn't freeze on jupyter notebook"""
    model = lstm.build_network([ncols, 150, 150, 1])
    model.fit_generator(
        data_gen_train,
        steps_per_epoch=steps_per_epoch,
        epochs=configs['model']['epochs']
    )
    model.save(configs['model']['filename_model'])
    print('> Model Trained! Weights saved in', configs['model']['filename_model'])
    return

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

# fix random seed for reproducibility
np.random.seed(7)

# 
# ***
# 1) load dataset
# ***
start_year = '2000'
end_year = '2016'
site = '0069'

# data_file_name = 'merged/max-merged_' + start_year + '-' + end_year + '.csv'
data_file_name = 'out/pca.csv'
df = read_csv(data_file_name, engine='python', skipfooter=3)
df = df.set_index(df.columns[0])
df.index.rename('id', inplace=True)

# remove_other_site_cols(df, site)

a = list(df)

for i in range (len(a)):
    print i, a[i]

# pick column to predict
# try:
#     target_col = int(raw_input("Select the column number to predict (default = " + a[len(a)-1] + "): "))
# except ValueError:
target_col = len(a)-1   #choose last column as default

# choose look-ahead to predict   
# try:
#     lead_time =  int(raw_input("How many days ahead to predict (default = 2)?: "))
# except ValueError:
lead_time = 4

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
# try:
#     process = raw_input("Does the data need to be pre-preprocessed Y/N? (default = y) ")
# except ValueError:
# process = 'y'
    
# if process == 'Y' or 'y':
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))

dataset = scalerX.fit_transform(dataset1)
dataplot = scalerY.fit_transform(dataplot1)
    
#     print'\nData processed using MinMaxScaler'
# else:
#     print'\nData not processed'
    
# split into train and test sets
splits = TimeSeriesSplit(n_splits=3)

# for train_index, test_index in splits.split(dataset)


train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

# prepare output arrays
# ***
# 3) dataplot[train_size:len(dataset)] changed because it should be dataplot len
# ***
trainY, testY = dataplot[0:train_size], dataplot[train_size:len(dataplot)]

n,p = np.shape(trainY)
if n < p:
    trainY = trainY.T
    testY = testY.T

# resize input sets
trainX1 = train[:len(trainY),]
testX1 = test[:len(testY),]
  
# get number of epochs
# try:
#     n_epochs = int(raw_input("Number of epochs? (Default = 10)? "))
# except ValueError:
n_epochs = 50
    
# prepare input Tensors
# try:
#     look_back = int(raw_input("Number of recurrent (look-back) units? (Default = 1)? "))
# except ValueError:
look_back = 2

trainX = TensorForm(trainX1, look_back)
testX = TensorForm(testX1, look_back)
# input_nodes = trainX.shape[2]

# ***
# 4) number of neuros / input_nodes increased for the LSTM layer
# ***
input_nodes = 150

# trim target arrays to match input lengths
if len(trainX) < len(trainY):
    trainY = np.asmatrix(trainY[:len(trainX)])
    
if len(testX) < len(testY):
    testY = np.asmatrix(testY[:len(testX)])

# model = lstm.build_network([ncols, 150, 150, 1])
# build lstm
model = Sequential()

model.add(LSTM(input_nodes,
    input_shape=(trainX.shape[1], trainX.shape[2]),
    return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(
    trainX.shape[2],
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation("tanh"))

start = time.time()
model.compile(
    loss=configs['model']['loss_function'],
    optimizer=configs['model']['optimiser_function'])

# t = threading.Thread(target=fit_model_threaded, args=[model, data_gen_train, steps_per_epoch, configs])
# t.start()

history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=512, validation_data=(testX, testY), shuffle=False)

# data_gen_test = dl.generate_clean_data(
#     configs['data']['filename_clean'],
#     batch_size=configs['data']['batch_size'],
#     start_index=ntrain
# )

# ntest = nrows - ntrain
# steps_test = int(ntest / configs['data']['batch_size'])
# print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

predictions = model.predict(testX)

#Save our predictions
# with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
#     dset_p = hf.create_dataset('predictions', data=predictions)
#     dset_y = hf.create_dataset('true_values', data=true_values)
    
# plot_results(predictions[:800], true_values[:800])
plot_results(predictions, testY)
#Reload the data-generator
# data_gen_test = dl.generate_clean_data(
#     configs['data']['filename_clean'],
#     batch_size=800,
#     start_index=ntrain
# )
# data_x, true_values = next(data_gen_test)
window_size = 50 #numer of steps to predict into the future

#We are going to cheat a bit here and just take the next 400 steps from the testing generator and predict that data in its whole
predictions_multiple = predict_sequences_multiple(
    model,
    testX   ,
    testX[0].shape[0],
    window_size
)

plot_results_multiple(predictions_multiple, testY, window_size)