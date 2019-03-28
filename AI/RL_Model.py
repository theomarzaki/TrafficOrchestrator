import torch
import torch.nn as nn

NUMBER_OF_ACTIONS = 5

class Dueling_DQN(nn.Module):
    def __init__(self):
        super(Dueling_DQN,self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUMBER_OF_ACTIONS)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage  - advantage.mean()

class DoubleQLearning(nn.Module):
    def __init__(self):
        super(DoubleQLearning, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUMBER_OF_ACTIONS)
        )

    def forward(self, x):
        return self.layers(x)

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(20,128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, NUMBER_OF_ACTIONS)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
