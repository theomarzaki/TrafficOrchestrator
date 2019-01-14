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
import logging

# ['globalXmerging' 'globalYmerging' 'lenghtMerging' 'widthMerging' 'velocityMerging' 'accelarationMerging' 'spacingMerging'
#  'globalXPreceding' 'globalYPreceding' 'lengthPreceding' 'widthPreceding' 'velocityPreceding' 'accelarationPreceding' 'globalXfollowing'
#  'globalYfollowing' 'widthFollowing' 'velocityFollowing''accelerationFollowing' 'spacingFollowing'] removed ['recommendation' 'heading'
#  'recommendedAcceleration']

class Agent():
    def __init__(self):
        pass

    def left_move(self,state):
        new_state = state

        return new_state

    def right_move(self,state):
        pass

    def accelerate_move(self,state):
        pass

    def deccelerate_move(self,state):
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv("csv/lineMergeDataWithHeading.csv")

num_epochs = 1
agent = Agent()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
train_data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1, inplace=True)
test_data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1, inplace=True)

featuresTrain = torch.zeros(math.ceil(train_data.shape[0]/70),70,19)

batch = torch.zeros(70,19)
counter = 0
for idx in range(train_data.shape[0]):
    if idx % 70 != 0 or idx == 0:
        batch[idx % 70]= torch.Tensor(train_data.values[idx])
    else:
        featuresTrain[counter] = batch
        counter = counter + 1
        batch = torch.zeros(70,19)


for epoch in range(num_epochs):
    for index,game_run in enumerate(featuresTrain):
        game_state = game_run
        if index == 0:
            for state in range(game_run.shape[0]):
                moves = {}
                moves["0"] = agent.accelerate_move(game_state[state])
                moves["1"] = agent.deccelerate_move(game_state[state])
                moves["2"] = agent.left_move(game_state[state])
                moves["3"] = agent.right_move(game_state[state])

                try:
                    pass
                    # game_state[state + 1] = torch.Tensor([0., 0., 0., 0., 0., 0., 0., state, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                except:
                    pass
