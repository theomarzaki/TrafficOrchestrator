# This script provides a way for the TO to predict the neighrest cars to the merging car to
# provide a state space for the lane merge enviroment. Uses an LSTM Model to acheive this:

# Sequential learning -> change to minibatch learning

# multi variable classifir -> sequence to sequence regressor

# @parameters input: Road state Tensor

# @parameters output: LSTM Model to predict next car states (preceeding,merging,following) jit trace file

# Created by: Omar Nassef(KCL)


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
from csv_data import Data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_wrapper = Data()
data = data_wrapper.get_data()

scaler_x = MinMaxScaler(feature_range =(0, 1))

train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
# train_data = scaler_x.fit_transform(train_data)
# test_data = scaler_x.transform(test_data)

featuresTrain,targetsTrain = data_wrapper.get_training_lstm_data()

featuresTest, targetsTest = data_wrapper.get_testing_lstm_data()

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=1,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=1,shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self,train_loader,test_loader):
        super(RNN, self).__init__()

        # Hyper-parameters
        self.sequence_length = 1
        self.input_size = 20
        self.hidden_size = 512
        self.num_layers = 3
        self.num_classes = 20
        self.batch_size = 32
        self.num_epochs = 1
        self.learning_rate = 0.1

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def train(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        hist = []
        total_step = len(self.train_loader)
        for epoch in range(self.num_epochs):
            for i, (locations, labels) in enumerate(self.train_loader):

                locations = locations.reshape(-1, self.sequence_length, self.input_size).to(device)
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
                           .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))

        return hist

    def test(self):
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

def main():
    model = RNN(train_loader,test_loader).to(device)

    hist = model.train()

    model.test()

    # Save the model checkpoint
    traced_script_module = torch.jit.trace(model, torch.rand(1,1,20))
    traced_script_module.save("lstm_model.pt")

    #
    # # scaler_x.transform(data)
    #
    # input = torch.zeros(1,1,20)
    #
    # input[0][0] = torch.Tensor(data.iloc[1])
    # # input[0][1] = torch.Tensor(data.iloc[2])
    #
    # x = model(input)
    #
    # print(x.data)
    #
    # # print(scaler_x.inverse_transform(x.data.numpy().reshape(1,-1)))
    #
    # print(data.iloc[2])

    plt.plot(hist)
    plt.show()

if __name__ == '__main__':
    main()
