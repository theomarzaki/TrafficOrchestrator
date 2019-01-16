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
from scipy.spatial import distance
import random
from torch.autograd import Variable

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
        self.EPSILON_DECAY = 100000

        self.fc1 = nn.Linear(1, 19)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(19, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(x.view(x.size()[0], -1))
        out = self.relu1(out)
        out = self.fc2(out)

        return out



# ['globalXmerging', 'globalYmerging', 'lenghtMerging',
#        'widthMerging', 'velocityMerging', 'accelarationMerging',
#        'spacingMerging', 'globalXPreceding', 'globalYPreceding',
#        'lengthPreceding', 'widthPreceding', 'velocityPreceding',
#        'accelarationPreceding', 'globalXfollowing', 'globalYfollowing',
#        'widthFollowing', 'velocityFollowing', 'accelerationFollowing',
#        'spacingFollowing', 'recommendation', 'heading',
#        'recommendedAcceleration'],

def isCarTerminal(state):
    y_diff = state[14] - state[8]
    x_diff = state[13] - state[7]
    slope = round(y_diff,2) / round(x_diff,2)
    plus_c = state[8] - (slope * state[7])

    if ((round(state[1]) + 2 == round(slope * state[0] + plus_c) or round(state[1]) - 2 == round(slope * state[0]))  and state[6] <= state[18]):
        return True; # C is on the line.
    return False;


def CalculateReward(state,predictor):
    reward = 0,False
    if predictor.predict_possible_merge(state) == False:
        reward = -1,False
    elif isCarTerminal(state) == True:
        reward = 1,True
    else:
        reward = 0.1,False

    return reward

class Agent():
    def __init__(self):
        self.accelerate_tensor = torch.Tensor([1,0,0,0,0])
        self.deccelerate_tensor = torch.Tensor([0,1,0,0,0])
        self.left_tensor = torch.Tensor([0,0,1,0,0])
        self.right_tensor = torch.Tensor([0,0,0,1,0])
        self.doNothing_tensor = torch.Tensor([0,0,0,0,1])

    def calculateActionComputed(self,action_tensor,state):
        if torch.equal(action_tensor,self.accelerate_tensor):
            return self.accelerate_move(state)
        elif torch.equal(action_tensor,self.deccelerate_tensor):
            return self.deccelerate_move(state)
        elif torch.equal(action_tensor,self.left_tensor):
            return self.left_move(state)
        elif torch.equal(action_tensor,self.right_tensor):
            return self.right_move(state)
        elif torch.equal(action_tensor,self.doNothing_tensor):
            return self.passive_move(state)
        else:
            logging.warning('inappropriate action -- SEE ME')

    def left_move(self,state):
        return state

    def right_move(self,state):
        return state

    def accelerate_move(self,state):
        return state

    def deccelerate_move(self,state):
        return state

    def passive_move(self,state):
        return state

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

epsilon = model.initial_epsilon

replay_memory = []
accelerate_tensor = torch.Tensor([1,0,0,0,0])
deccelerate_tensor = torch.Tensor([0,1,0,0,0])
left_tensor = torch.Tensor([0,0,1,0,0])
right_tensor = torch.Tensor([0,0,0,1,0])
doNothing_tensor = torch.Tensor([0,0,0,0,1])


for epoch in range(num_epochs):
    for index,game_run in enumerate(featuresTrain):
        game_state = game_run
        if index < 2:
            for current_epoch in range(game_run.shape[0]):
                for state in range(game_state.shape[0]):
                    current = game_state[state].data.cpu().numpy()
                    output = model(torch.Tensor(current))[0]

                    # initialise actions

                    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
                    random_action = random.random() <= epsilon
                    if random_action: print("Performed random action!")
                    action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                                    if random_action
                                    else torch.argmax(output)][0]

                    action[action_index] = 1

                    # get next state and reward

                    next_state = agent.calculateActionComputed(action,current)
                    reward,terminal = CalculateReward(next_state,predictor)

                    replay_memory.append((torch.Tensor(current), torch.Tensor(action), reward, torch.Tensor(next_state), terminal))

                    # if replay memory is full, remove the oldest transition
                    if len(replay_memory) > model.replay_memory_size:
                        replay_memory.pop(0)

                    epsilon = model.final_epsilon + (model.initial_epsilon - model.final_epsilon) * \
                                     math.exp(-1. * current_epoch / model.EPSILON_DECAY)

                    minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

                    # minibatch = torch.Tensor(minibatch)
                    # unpack minibatch
                    current_batch = Variable(torch.cat(tuple(d[0] for d in minibatch)))
                    action_batch = Variable(torch.Tensor(tuple(d[1] for d in minibatch)[0]))
                    reward_batch = Variable(torch.Tensor(tuple(d[2] for d in minibatch)))
                    next_state_batch = Variable(torch.cat(tuple(d[3] for d in minibatch)))
                    terminal_state_batch = tuple(d[4] for d in minibatch)

                    # get output for the next state
                    next_state_batch_output = model(next_state_batch)

                    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                    y_batch = tuple(reward_batch[i] if minibatch[i][4]
                                        else reward_batch[i] + model.gamma * torch.max(next_state_batch_output[i])
                                              for i in range(len(minibatch)))

                    # extract Q-value
                    q_value = torch.sum(model(current_batch) * action_batch, dim=1)

                    # PyTorch accumulates gradients by default, so they need to be reset in each pass
                    optimizer.zero_grad()

                    # returns a new Tensor, detached from the current graph, the result will never require gradient
                    y_batch = torch.Tensor(y_batch).detach()

                    print(q_value)
                    print(y_batch)

                    # calculate loss
                    loss = criterion(q_value, y_batch)

                    print('Loss:   {:.4f}'.format(loss.item()))

                    # do backward pass
                    loss.backward()
                    optimizer.step()

                    print(next_state)
                    if terminal == True:
                        break
                    else:
                        game_state[state + 1] = torch.Tensor(next_state)
