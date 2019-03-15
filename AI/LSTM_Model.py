# This script provides a way for the TO to predict the neighrest cars to the merging car to
# provide a state space for the lane merge enviroment. Uses an LSTM Model to acheive this:

# Sequential learning -> change to minibatch learning

# multi variable classifir -> sequence to sequence regressor

# @parameters input: Road state Tensor

# @parameters output: LSTM Model to predict next car states (preceeding,merging,following) jit trace file

# Created by: Omar Nassef(KCL)

#   -- BENCHMARK --
#
#   CONFIG :
#     sequence_length = 1
#     input_size = 13
#     hidden_size = 128
#     num_layers = 2
#     num_classes = 13
#     batch_size = 32
#     learning_rate = 1e-5
#
#   + [CPU-MODE] i7 6700K Stock + 16 Go DDR4 2166 Non-ECC RAM
#    |=> 0.12 Epoch/s
#
#   + [CPU-MODE] i5 3570K Stock + 16 Go DDR3 1600 Non-ECC RAM
#    |=> 0.08 Epoch/s
#
#   + [GPU-MODE] GTX 970 Stock + i5 3570K Stock + 16 Go DDR3 1333 Non-ECC RAM
#    |=> 0.58 Epoch/s | 52% + 1050MB uses with CPU bottleneck, so I think I can push a lot further. Maybe 1 Epoch/s
#

import time
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
from csv_data import Data

# Device configuration
loader_batch_size = 128
view_rate = 128
dev_type = 'cpu'

num_epochs = 5000

sequence_length = 1
input_size = 13
hidden_size = 128
num_layers = 2
num_classes = 13
batch_size = 3
learning_rate = 1e-7

if torch.cuda.is_available():
    dev_type = 'cuda'
    view_rate /= 4
    loader_batch_size *= 8
    print("NVIDIA GPU detected and use !")
else: print("/!\ No NVIDIA GPU detected, we stick to the CPU !")

device = torch.device(dev_type)

data_wrapper = Data()
data = data_wrapper.get_data()

featuresTrain,targetsTrain = data_wrapper.get_training_lstm_data()
featuresTest, targetsTest = data_wrapper.get_testing_lstm_data()

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=loader_batch_size,shuffle=False,
                                           num_workers = 3, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=loader_batch_size,shuffle=False,
                                          num_workers = 3, pin_memory=True)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self,train_loader,test_loader):
        super(RNN, self).__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def init_hidden(self):
    # This is what we'll initialise our hidden state as
        return (torch.zeros(num_layers, batch_size, hidden_size),
            torch.zeros(num_layers, batch_size, hidden_size))

    def init_hidden(self):
    # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train(model):
    print("Go !")
    criterion = nn.MSELoss()
    current_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    hist = []
    total_step = len(model.train_loader)
    for epoch in range(num_epochs):
        model.zero_grad()
        # model.hidden = model.init_hidden()
        loss = None
        for i, (locations, labels) in enumerate(model.train_loader):

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

            if (i+1) % view_rate == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        bufftime = time.time()
        delta_time = bufftime - current_time
        print("+ "+str(delta_time)+" Secs")
        print("Perf: "+str(1/delta_time)+" Epoch/s\n")
        current_time = bufftime

    return hist

def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for locations, labels in test_loader:
            locations = locations.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(locations)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            for idx, actual in enumerate(locations):
                if ((actual == outputs.data).all()):
                    correct += 1

            # correct += (predicted.long() == labels.long()).sum().item()
            
        print('Test Accuracy of the model of the model: {} %'.format(100 * correct / total))

def main():
    model = RNN(train_loader,test_loader).to(device,non_blocking=True)

    hist = train(model)

    test(model)

    traced_script_module = torch.jit.trace(model, torch.rand(1,1,input_size).to(device))
    traced_script_module.save("lstm_model.pt")

    plt.plot(hist)
    plt.show()

if __name__ == '__main__':
    main()
