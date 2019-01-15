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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import sklearn.neural_network
from sklearn.ensemble import RandomForestRegressor
import logging



class RandomForestPredictor():

    def __init__(self):
        self.data = data
        self.train_and_test_random_forest_classifier()

    def countClassifiedResults(self,predictions):
        cnt = 0
        for i in range(len(predictions)):
            if predictions[i] != 1:
                cnt += 1

        return cnt, len(predictions)-cnt

    def train_and_test_random_forest_classifier(self):
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=1)
        self.train, self.validation = train_test_split(self.train_data, test_size=0.2, random_state=1)

        self.X_trainRecommendation = self.train.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1)
        self.Y_trainRecommendation = self.train['recommendation']

        self.X_valRecommendation = self.validation.drop(["recommendation", 'heading', 'recommendedAcceleration'], axis=1).copy()
        self.Y_valRecommendation = self.validation['recommendation']

        self.X_testRecommendation = self.test_data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1)
        self.Y_testRecommendation = self.test_data['recommendation']

        self.random_forest = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1)

        self.random_forest.fit(self.X_trainRecommendation, self.Y_trainRecommendation)

        acc_random_forest = round(self.random_forest.score(self.X_trainRecommendation, self.Y_trainRecommendation) * 100, 2)
        print("Training accuracy: ", acc_random_forest)

        self.Y_pred = self.random_forest.predict(self.X_valRecommendation)

        print(self.countClassifiedResults(self.Y_pred))

        acc_random_forest_val = round(accuracy_score(self.Y_valRecommendation, self.Y_pred)*100, 2)
        print("Validation accuracy: ", acc_random_forest_val)

    def predict_possible_merge(self,prediction_variables):
        # try:
        prediction_array = self.random_forest.predict(prediction_variables.reshape(1,-1))
        if len(prediction_array) != 0:
            if prediction_array[0] == 1:
                return True
            else:
                return False
        else:
            logging.warning('Trying to find predictions in an empty array')
        # except:
        #     logging.error('Data is not in Correct Format for Train')

class DeepQLearning(nn.Module):
    def __init__(self):
        super(DeepQLearning,self).__init__()

        self.number_of_actions = 5
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32 #TODO may need to change this

        self.fc1 = nn.Linear(1, 19)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(19, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out



# ['globalXmerging' 'globalYmerging' 'lenghtMerging' 'widthMerging' 'velocityMerging' 'accelarationMerging' 'spacingMerging'
#  'globalXPreceding' 'globalYPreceding' 'lengthPreceding' 'widthPreceding' 'velocityPreceding' 'accelarationPreceding' 'globalXfollowing'
#  'globalYfollowing' 'widthFollowing' 'velocityFollowing''accelerationFollowing' 'spacingFollowing'] removed ['recommendation' 'heading'
#  'recommendedAcceleration']

def CalculateReward(state,predictor):
    if predictor.predict_possible_merge(state) == True:
        print("well")
    else:
        print("damn")


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



predictor = RandomForestPredictor()

#TRAIN RL
model = DeepQLearning()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()

replay_memory = []
accelerate_tensor = torch.Tensor([1,0,0,0,0])
deccelerate_tensor = torch.Tensor([0,1,0,0,0])
left_tensor = torch.Tensor([0,0,1,0,0])
right_tensor = torch.Tensor([0,0,0,1,0])
doNothing_tensor = torch.Tensor([0,0,0,0,1])


for epoch in range(num_epochs):
    for index,game_run in enumerate(featuresTrain):
        game_state = game_run
        if index == 0:
            for state in range(game_run.shape[0]):
                current_state = state
                moves = {}
                moves["0"] = agent.accelerate_move(game_state[current_state])
                moves["1"] = agent.deccelerate_move(game_state[current_state])
                moves["2"] = agent.left_move(game_state[current_state])
                moves["3"] = agent.right_move(game_state[current_state])
                moves["4"] = game_state[current_state]

                CalculateReward(game_state[current_state].data.cpu().numpy(),predictor)

                try:
                    pass
                    # game_state[state + 1] = torch.Tensor([0., 0., 0., 0., 0., 0., 0., state, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                except:
                    pass
