"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from csv_data import Data
import torchvision

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 20      # rnn input size
LR = 0.001           # learning rate


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=128,     # rnn hidden unit
            num_layers=2,       # number of rnn layer
            # batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(128, 12)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

data_wrapper = Data()
data = data_wrapper.get_data()

featuresTrain,targetsTrain = data_wrapper.get_training_lstm_data()

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=32,shuffle=True)

for epoch in range(100):
    for i, (locations, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        locations = locations.reshape(-1, 1, 20)
        labels = labels

        # Forward pass
        outputs, h_state = rnn(locations,h_state)

        h_state = h_state.data

        loss = loss_func(outputs, labels)

        # Backward and optimize

        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, 100, i+1, len(train_loader), loss.item()))
