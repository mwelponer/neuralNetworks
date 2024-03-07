import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


"""
Generate a set of random temperatures using a sinusoidal signal
"""
def genSinTemperatures(num_samples, min_temp, max_temp, frequency, noise_std, plot=False):
    # generate a Time vector
    time = np.linspace(0, 100, num_samples) 

    # generate sinusoidal signal
    #frequency = 0.2 # 0.05 # frequency in Hz
    amplitude = (max_temp - min_temp) / 2 # half of temperature range
    offset = (max_temp + min_temp) / 2 # mid-range to shift signal along y 
    signal = amplitude * np.sin(2 * np.pi * frequency * time) + offset
    # print(signal)

    # generate random noise
    noise = np.random.normal(-noise_std, noise_std, num_samples)  # mean=0, std=noise_std
    # print(noise)

    # combine sinusoidal signal with noise
    temperature = signal + noise

    # plot data 
    if plot:
        plt.plot(time, temperature, label='Temperature')
        plt.title('Sinusoidal Random Temperatures')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.grid(True)
        plt.show()

    return temperature


"""
convert an array of values into a dataset matrix
"""
def createDataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		dataX.append(dataset[i:(i+look_back), 0])
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
#tf.random.set_seed(7)

# # load the dataset
# df = pd.read_csv('data/passengers.csv', usecols=[1], engine='python')
# dataset = df.values
# dataset = dataset.astype('float32')
###### generate random temperatures
dataset = genSinTemperatures(num_samples=500, min_temp=25, max_temp=35, \
                          frequency=0.3, noise_std=1.5, plot=True)
dataset = np.reshape(dataset, [-1, 1])
# print(type(dataset))
# print(dataset[:10])
# exit()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# print(dataset[:10])
# exit()

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = createDataset(train, look_back)
testX, testY = createDataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()