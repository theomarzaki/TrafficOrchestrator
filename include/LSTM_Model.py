import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
import math
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 1
input_size = 19
hidden_size = 200
num_layers = 1
num_classes = 19
batch_size = 1
num_epochs = 1
learning_rate = 0.01

data = pd.read_csv("csv/lineMergeDataWithHeading.csv")

scaler_x = MinMaxScaler(feature_range =(0, 1))

train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
train_data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1, inplace=True)
test_data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1, inplace=True)
# train_data = scaler_x.fit_transform(train_data)
# test_data = scaler_x.transform(test_data)

featuresTrain = torch.zeros(math.ceil(train_data.shape[0]/2),1,19)
targetsTrain = torch.zeros(math.ceil(train_data.shape[0]/2),19)

featuresTest = torch.zeros(math.ceil(test_data.shape[0]/2),1,19)
targetsTest= torch.zeros(math.ceil(test_data.shape[0]/2),19)

batch = torch.zeros(1,19)
for idx in range(train_data.shape[0]):
    if idx % 2 != 0:
        batch[0]= torch.Tensor(train_data.values[idx])
    else:
        pos = math.ceil(idx/2)
        featuresTrain[pos] = batch
        targetsTrain[pos] = torch.Tensor(train_data.values[idx])
        batch = torch.zeros(1,19)

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

batch = torch.zeros(1,19)
for idx in range(test_data.shape[0]):
    if idx % 2 != 0:
        batch[0]= torch.Tensor(test_data.values[idx])
    else:
        pos = math.ceil(idx/2)
        featuresTest[pos] = batch
        targetsTest[pos] = torch.Tensor(test_data.values[idx])
        batch = torch.zeros(1,19)


test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
hist = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (locations, labels) in enumerate(train_loader):

        locations = locations.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(locations)

        loss = criterion(outputs, labels)
        hist.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for locations, labels in test_loader:
        locations = locations.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(locations)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        for idx,actual in enumerate(locations):
            if((actual == outputs.data).all()):
                correct += 1

        # correct += (predicted.long() == labels.long()).sum().item()

    print('Test Accuracy of the model of the model: {} %'.format(100 * correct / total))

# Save the model checkpoint
traced_script_module = torch.jit.trace(model, torch.rand(1,2,19))
traced_script_module.save("lstm_model.pt")

data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1, inplace=True)
# scaler_x.transform(data)

input = torch.zeros(1,1,19)

input[0][0] = torch.Tensor(data.iloc[1])
# input[0][1] = torch.Tensor(data.iloc[2])

x = model(input)

print(x.data)

# print(scaler_x.inverse_transform(x.data.numpy().reshape(1,-1)))

print(data.iloc[2])

plt.plot(hist)
plt.show()
