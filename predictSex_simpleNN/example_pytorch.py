# pip install -U scikit-learn
# pip3 install torch torchvision torchaudio

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print(f"pytorch version: {torch.__version__}")

EPOCH_NUMBER = 2000


############### SAMPLE DATA
# Sample data [height, weight, footsize] and labels 1 male, 0 female
# samples = [[150, 67, 37], [130, 60, 34], [200, 65, 46], [125, 52, 37], \
#    [210, 82, 46], [181, 70, 43]]
# labels = [0, 0, 1, 0, 1, 1]
from genData import labels, samples # randomply create 
x = torch.tensor(samples, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1) 
# view -1, 1 means guess elements for rows knowing that elements for cols is 1
# so it will be [[0], [0], [1], [0], [1], [1]]


############### DEFINE THE NEURAL NETWORK
class NeuralNet(nn.Module): # extend nn.Module
    def __init__(self):
        super(NeuralNet, self).__init__()
        # input layer of 3 neurons (height, weight, footsize)
        self.fc1 = nn.Linear(3, 16)  # first hidden layer (16 neurons) 
        self.fc2 = nn.Linear(16, 32) # second hidden layer (32 neurons)
        self.fc3 = nn.Linear(32, 1)  # ouput layer (1 neuron)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


############### INSTANTIATE THE NETWORK 
model = NeuralNet()

# Define LOSS FUNCTION and BACKPROPAGATION (OPTIMIZER)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss 
#optimizer = optim.SGD(model.parameters(), lr=0.01) # Stocastic gradient descend SGD
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam


############### TRAIN THE NEURAL NETWORK
# Lists to store accuracy and loss values
loss_values = []
accuracy_values = []

for epoch in range(EPOCH_NUMBER):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Compute accuracy
    predicted = torch.round(outputs)
    accuracy = (predicted == y).sum().item() / len(y)

    # Backward pass and optimization
    optimizer.zero_grad() # Resets the gradients of all optimized
    loss.backward() # apply loss function
    optimizer.step() # update weights  

    # Append accuracy and loss values
    loss_values.append(loss.item())
    accuracy_values.append(accuracy)

    if (epoch+1) % (EPOCH_NUMBER/10) == 0:
        print(f'Epoch [{epoch+1}/{EPOCH_NUMBER}], Loss: {loss.item():.4f}')


############### PLOT ACCURACY AND LOSS
plt.figure(figsize=(10, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCH_NUMBER + 1), loss_values, label='Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCH_NUMBER + 1), accuracy_values, label='Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


############### PREDICT ON TRAINING DATA
with torch.no_grad(): # during inference disable gradient claculation (save memory)
    predicted = model(x)
    predicted_binary = np.round(predicted.numpy())  # Rounding to get 0 or 1

accuracy = accuracy_score(y.flatten().numpy(), predicted_binary)
precision = precision_score(y.flatten().numpy(), predicted_binary)
recall = recall_score(y.flatten().numpy(), predicted_binary)
f1 = f1_score(y.flatten().numpy(), predicted_binary)
# Print evaluation metrics
print("\nAccuracy:", accuracy) # model correctness
print("Precision:", precision) # ability to avoid false positives
print("Recall:", recall) # ability to predict true positives
print("F1 Score:", f1) # harmonic mean of precision and recall


############### PREDICT ON TEST SET
test = [[150, 60, 37], [155, 47, 38], [182, 70, 43], [190, 80, 45], [172, 63, 42]]
test_pt = torch.tensor(test, dtype=torch.float32)

with torch.no_grad():
    predicted = model(test_pt)
    #predicted = np.round(predicted.numpy())
#print(f"Predict {test}: {predicted.flatten()}")

    # Convert predictions to sex labels
    sex_labels = ['Female' if pred < 0.5 else 'Male' for pred in predicted.numpy()]
    print("\nPredictions:", sex_labels)


