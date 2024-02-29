# pip install tensorflow
# pip install -U scikit-learn

from keras.models import Sequential
from keras.layers import Dense
#from keras import regularizers
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


import keras, tensorflow
print(f"keras version: {keras.__version__}")
print(f"tensorflow version: {tensorflow.__version__}")


EPOCH_NUMBER = 100


############### SMAPLE DATA 
# Sample data [height, weight, footsize] and labels 1 male, 0 female
# samples = [[150, 67, 37], [130, 60, 34], [200, 65, 46], [125, 52, 37], \
#     [210, 82, 46], [181, 70, 43]]
# labels = [0, 0, 1, 0, 1, 1]
from genData import labels, samples # randomply create 
x, y = np.array(samples), np.array(labels)


############### DEFINE AND COMPILE THE MODEL
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(32, activation='relu'), #, kernel_initializer='glorot_uniform'), # Xavier init
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])


############### TRAIN THE MODEL: TWO OPTIONS
# 1. use all samples for training, i.e. no validation.
# Prediction is done just on training data, at the end of each epoch
history = model.fit(x, y, batch_size=20, epochs=EPOCH_NUMBER, shuffle=True)

# 2. validation_split uses 10% as validation, i.e. predict on training 
# data (loss/accuracy) and on validation set (valLoss/valAccuracy)
#history = model.fit(x, y, batch_size=20, validation_split=.10, epochs=EPOCH_NUMBER, shuffle=True) 


############### PLOT ACCURACY AND LOSS
# Plot training history
plt.figure(figsize=(10, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training [and Validation] Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training [and Validation] Loss')
plt.legend()

plt.show() 


############### PREDICT (AGAIN ALL IN ONE TIME) ON TRAINING DATA
# True labels
y_true = y
# Predicted labels
y_pred = np.round(model.predict(x)).astype(int)
# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("\nAccuracy:", accuracy) # model correctness
print("Precision:", precision) # ability to avoid false positives
print("Recall:", recall) # ability to predict true positives
print("F1 Score:", f1) # harmonic mean of precision and recall
# confusion matrix is made like this
# [ [true positives, false positives],
#   [false negatives, true negatives] ]
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


############### PREDICT ON TEST SET
test = [[150, 60, 37], [155, 47, 38], [182, 70, 43], [190, 80, 45], [172, 63, 42]]
predictions = model.predict(np.array(test))

# Convert predictions to sex labels
sex_labels = ['Female' if pred < 0.5 else 'Male' for pred in predictions]
print("\nPredictions:", sex_labels)

# predictions = np.round(predictions)
# print(f"Predict {test}: {predictions.flatten()}")