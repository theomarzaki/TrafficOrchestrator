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
        self.replay_memory_size = 50
        self.minibatch_size = 32 #TODO may need to change this
        self.EPSILON_DECAY = 100000

        self.fc1 = nn.Linear(19,200)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(200, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)

        return out



# ['globalXmerging', 'globalYmerging', 'lenghtMerging',
#        'widthMerging', 'velocityMerging', 'accelarationMerging',
#        'spacingMerging', 'globalXPreceding', 'globalYPreceding',
#        'lengthPreceding', 'widthPreceding', 'velocityPreceding',
#        'accelarationPreceding', 'globalXfollowing', 'globalYfollowing',
#        'widthFollowing', 'velocityFollowing', 'accelerationFollowing',
#        'spacingFollowing',] 'recommendation', 'heading',
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
        displacement = state[4] * 0.001 + 0.5 * (state[5] * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position - (0.001 * new_position)
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        return new_state

    def right_move(self,state):
        displacement = state[4] * 0.001 + 0.5 * (state[5] * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position + (0.001 * new_position)
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        return new_state

    def accelerate_move(self,state):
        final_velocity = state[4] + 0.001 * (state[4] + state[5] * 0.001)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.001)
        displacement = final_velocity * 0.001 + 0.5 * (final_acceleration * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        new_state[4] = final_velocity
        new_state[5] = final_acceleration
        return new_state

    def deccelerate_move(self,state):
        final_velocity = state[4] - 0.001 * (state[4] + state[5] * 0.001)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.001)
        displacement = final_velocity * 0.001 + 0.5 * (final_acceleration * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        new_state[4] = final_velocity
        new_state[5] = final_acceleration
        return new_state

    def passive_move(self,state):
        displacement = state[4] * 0.001 + 0.5 * (state[5] * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        return new_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv("csv/lineMergeDataWithHeading.csv")
data = data[::-1]


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

for epoch in range(num_epochs):
    for index,game_run in enumerate(featuresTrain):
        game_state = game_run
        if index < 2:
            for current_epoch in range(game_run.shape[0]):
                for state in range(game_state.shape[0]):
                    current = game_state[state].data.cpu().numpy()
                    try:
                        s_next = game_state[state + 1].data.cpu().numpy()
                    except:
                        pass
                    output = model(torch.from_numpy(current))
                    # initialise actions

                    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
                    random_action = random.random() <= epsilon
                    action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                                    if random_action
                                    else torch.argmax(output)][0]

                    action[action_index] = 1

                    # get next state and reward

                    next_state = agent.calculateActionComputed(action,current)
                    s_next[0] = next_state[0]
                    s_next[1] = next_state[1]
                    s_next[2] = next_state[2]
                    s_next[3] = next_state[3]
                    s_next[4] = next_state[4]
                    s_next[5] = next_state[5]
                    s_next[6] = next_state[6]
                    reward,terminal = CalculateReward(next_state,predictor)


                    # if replay memory is full, remove the oldest transition
                    if len(replay_memory) > model.replay_memory_size:
                        replay_memory.pop(0)

                    replay_memory.append((torch.Tensor(current), torch.Tensor(action), reward, torch.Tensor(next_state), terminal))


                    epsilon = model.final_epsilon + (model.initial_epsilon - model.final_epsilon) * \
                                     math.exp(-1. * current_epoch / model.EPSILON_DECAY)

                    minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

                    current_batch = torch.zeros(len(minibatch),19)
                    action_batch = torch.zeros(len(minibatch),5)
                    reward_batch = torch.zeros(len(minibatch))
                    next_state_batch = torch.zeros(len(minibatch),19)
                    terminal_state_batch = []
                    for idx,data_point in enumerate(minibatch):
                        current_batch[idx] = data_point[0]
                        action_batch[idx] = data_point[1]
                        reward_batch[idx] = data_point[2]
                        next_state_batch[idx] = data_point[3]
                        terminal_state_batch.append(data_point[4])

                    next_state_batch_output = torch.zeros(32,5)
                    for idx in range(next_state_batch.shape[0]):
                        next_state_batch_output[idx] = model(torch.Tensor(next_state_batch[idx]))[0]

                    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                    y_batch = tuple(reward_batch[i] if terminal_state_batch[i]
                                        else reward_batch[i] + model.gamma * torch.max(next_state_batch_output[i])
                                              for i in range(len(minibatch)))

                    # extract Q-value
                    q_value = torch.sum(model(current_batch) * action_batch, dim=1)

                    # PyTorch accumulates gradients by default, so they need to be reset in each pass
                    optimizer.zero_grad()

                    # returns a new Tensor, detached from the current graph, the result will never require gradient
                    y_batch = torch.Tensor(y_batch).detach()

                    # calculate loss
                    loss = criterion(q_value, y_batch)

                    if state % 10 == 0:
                        print('Epoch: {}, Runs: {}, Loss:   {:.4f}'.format(epoch,index,loss.item()))

                    # do backward pass
                    loss.backward()
                    optimizer.step()

                    if terminal == True:
                        break
                    else:
                        try:
                            game_state[state + 1] = torch.Tensor(next_state)
                        except:
                            print("no more states (time) for maneuvers")
                            break


traced_script_module = torch.jit.trace(model, torch.rand(19))
traced_script_module.save("rl_model.pt")
