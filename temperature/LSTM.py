from keras.models import Sequential
from keras.layers import Dense, LSTM
#from keras import regularizers
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import utils


NUM_TRAINING_SAMPLES = 750
NOISE_STD = 2.5
TESTING_PERCENTAGE = .30
NUM_EPOCHS = 50 # 50
LEARNING_RATE = 0.001 # 0.001
TIME_STEPS = 3


###### generate random temperatures
temperature = utils.genSinTemperatures(num_samples=NUM_TRAINING_SAMPLES, \
        min_temp=25, max_temp=35, frequency=0.3, noise_std=NOISE_STD, plot=True)

###### normalize temperatures [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
temperature = np.array(temperature) # convert to np array
temperature = temperature.reshape(-1, 1) # reshape to n rows, 1 column
temperature = scaler.fit_transform(temperature)

###### prepare data for LSTM
# # sample x = actual day, label y = next day
# samples = temperature[0:-1] # from 0 to n-1
# labels = temperature[1:] # from 1 to n
X, y = utils.prepareLSTN_Data(temperature, Nsteps=TIME_STEPS)
# reshape to (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

###### Split data into training and testing sets (testing set is still labelled)
X_train, y_train, X_test, y_test = utils.splitTrainingData(X, y, \
            testingPercentage=TESTING_PERCENTAGE)

###### define and compile a network model
model = Sequential()
model.add(LSTM(3, input_shape=(TIME_STEPS, 1)))
model.add(Dense(1))
# compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])

###### train the model
#history = model.fit(samples, labels, epochs=100, batch_size=1)
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=1, \
                    validation_split=.10, verbose=2)

###### Plot training history
plt.figure(figsize=(10, 5))
# Plot mae
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Mean Absolute Error')
plt.plot(history.history['val_mae'], label='Validation Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training/Validation Mean Absolute Error')
plt.legend()
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Validation Loss')
plt.legend()

plt.show() 

###### Predict temperatures for testing set
y_pred = model.predict(X_test)

###### de-normalize predictions
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

###### Plot predicted vs true temperatures for the testing set
plt.plot(y_test, label='True Temperatures')
plt.plot(y_pred, label='Predicted Temperatures')
plt.title('Predicted vs True Temperatures (LSTM)')
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()