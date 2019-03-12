# This Script provides a way to approximate the best actions for the agent to undertake
# in order lane merge

# Contains Random Forest Classifer to assign rewards to agent.

# @parameters input: LSTM output (enviroment state)

# @parameters output: RL Model (argmax) jit trace file

# Created by: Omar Nassef(KCL)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from RandomForestClassifier import RandomForestPredictor
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
from Agent import Agent
from csv_data import Data
from utils import CalculateReward,isCarTerminal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepQLearning(nn.Module):
    def __init__(self):
        super(DeepQLearning,self).__init__()

        self.learn_step_counter = 0
        self.num_epochs = 10000
        self.number_of_actions = 5
        self.gamma = 0.90
        self.final_epsilon = 0.01
        self.initial_epsilon = 1.0
        self.replay_memory_size = 10000 #may need to increase
        self.minibatch_size = 32 #TODO may need to change this
        self.EPSILON_DECAY = 100000

        self.fc1 = nn.Linear(20,512)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def train(self,model,target,featuresTrain,agent,predictor):
        replay_memory = []
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        epsilon = model.initial_epsilon
        counter = 0
        for epoch in range(self.num_epochs):
            for index,game_run in enumerate(featuresTrain):
                game_state = game_run
                counter = counter + 1
                for state in range(game_state.shape[0]):
                    current = game_state[state].data.cpu().numpy()
                    try:
                        s_next = game_state[state + 1].data.cpu().numpy()
                    except:
                        pass

                    if self.learn_step_counter % 100 == 0:
                        target.load_state_dict(model.state_dict())
                        self.learn_step_counter += 1

                    output = model(torch.from_numpy(current).to(device)).to(device)
                    # initialise actions

                    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
                    random_action = random.random() <= epsilon
                    action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                                    if random_action
                                    else torch.argmax(output)][0]

                    action[action_index] = 1

                    # get next state and reward

                    next_state = agent.calculateActionComputed(action,current,s_next)

                    reward,terminal = CalculateReward(next_state,predictor)

                    # if replay memory is full, remove the oldest transition
                    if len(replay_memory) > model.replay_memory_size:
                        replay_memory.pop(0)


                    replay_memory.append((torch.Tensor(current).to(device), torch.Tensor(action).to(device), reward, torch.Tensor(next_state).to(device), terminal))


                    epsilon = model.final_epsilon + (model.initial_epsilon - model.final_epsilon) * \
                                     math.exp(-1. * self.learn_step_counter / model.EPSILON_DECAY)

                    minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

                    current_batch = torch.zeros(len(minibatch),20).to(device)
                    action_batch = torch.zeros(len(minibatch),5).to(device)
                    reward_batch = torch.zeros(len(minibatch)).to(device)
                    next_state_batch = torch.zeros(len(minibatch),20).to(device)
                    terminal_state_batch = []
                    for idx,data_point in enumerate(minibatch):
                        current_batch[idx] = data_point[0]
                        action_batch[idx] = data_point[1]
                        reward_batch[idx] = data_point[2]
                        next_state_batch[idx] = data_point[3]
                        terminal_state_batch.append(data_point[4])

                    next_state_batch_output = torch.zeros(self.minibatch_size,5).to(device)
                    for idx in range(next_state_batch.shape[0]):
                        next_state_batch_output[idx] = model(next_state_batch[idx]).to(device)[0]


                    # No use of Target Network
                    # # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                    # y_batch = tuple(reward_batch[i] if terminal_state_batch[i]
                    #                     else reward_batch[i] + model.gamma * torch.max(next_state_batch_output[i])
                    #                           for i in range(len(minibatch)))
                    #
                    # # extract Q-value
                    # q_value = torch.sum(model(current_batch) * action_batch, dim=1)

                    # Use of Target Network
                    q_eval = torch.sum(model(current_batch).to(device) * action_batch, dim=1)  # shape (batch, 1)
                    q_next = target(next_state_batch).to(device).detach()     # detach from graph, don't backpropagate
                    q_target = reward_batch + 0.9 * q_next.max(1)[0]   # shape (batch, 1)
                    loss = criterion(q_eval, q_target)



                    # PyTorch accumulates gradients by default, so they need to be reset in each pass
                    optimizer.zero_grad()

                    # No use of Target Network
                    # returns a new Tensor, detached from the current graph, the result will never require gradient
                    # y_batch = torch.Tensor(y_batch).detach()

                    # calculate loss
                    # loss = criterion(q_value, y_batch)

                    if(self.learn_step_counter % 500 == 0):
                        model_state = {
                            'epoch': counter,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss':loss,
                            }
                        model_save_name = F"DQN{counter}.tar"
                        path = F"DQN_Saves/{model_save_name}"
                        torch.save(model_state, path)

                    if(state % 70 == 0):
                        print('Epoch: {}/{},Runs: {}/{}, Loss: {:.4f}, Average Reward: {:.2f}'.format(epoch,self.num_epochs,index,featuresTrain.shape[0],loss.item(),sum(reward_batch)/self.minibatch_size))

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


def main():

    data_wrapper = Data()
    data = data_wrapper.get_data()

    agent = Agent()

    featuresTrain = data_wrapper.get_training_data_tensor()

    predictor = RandomForestPredictor(data_wrapper.get_RFC_dataset())

    #TRAIN RL
    model = DeepQLearning().to(device)
    target_network = DeepQLearning().to(device)

    state = torch.load('rl_classifier.tar',map_location='cpu')
    model.load_state_dict(state['state_dict'])

    # model.train(model,target_network,featuresTrain,agent,predictor)

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save("rl_model.pt")

if __name__ == '__main__':
    main()