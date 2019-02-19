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


class DeepQLearning(nn.Module):
    def __init__(self):
        super(DeepQLearning,self).__init__()

        self.learn_step_counter = 0
        self.num_epochs = 500
        self.number_of_actions = 5
        self.gamma = 0.99
        self.final_epsilon = 0.01
        self.initial_epsilon = 1.0
        self.replay_memory_size = 10000 #may need to increase
        self.minibatch_size = 32 #TODO may need to change this
        self.EPSILON_DECAY = 50

        self.fc1 = nn.Linear(20,512)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def train(self,model,target,featuresTrain,agent):
        replay_memory = []
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        criterion = nn.MSELoss()
        epsilon = model.initial_epsilon
        for epoch in range(self.num_epochs):
            for index,game_run in enumerate(featuresTrain):
                game_state = game_run
                for current_epoch in range(game_run.shape[0]):
                    for state in range(game_state.shape[0]):
                        current = game_state[state].data.cpu().numpy()
                        try:
                            s_next = game_state[state + 1].data.cpu().numpy()
                        except:
                            pass

                        if self.learn_step_counter % 100 == 0:
                            target.load_state_dict(model.state_dict())
                            self.learn_step_counter += 1


                        output = model(torch.from_numpy(current))
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

                        replay_memory.append((torch.Tensor(current), torch.Tensor(action), reward, torch.Tensor(next_state), terminal))


                        epsilon = model.final_epsilon + (model.initial_epsilon - model.final_epsilon) * \
                                         math.exp(-1. * current_epoch / model.EPSILON_DECAY)

                        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

                        current_batch = torch.zeros(len(minibatch),20)
                        action_batch = torch.zeros(len(minibatch),5)
                        reward_batch = torch.zeros(len(minibatch))
                        next_state_batch = torch.zeros(len(minibatch),20)
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


                        # No use of Target Network
                        # # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                        # y_batch = tuple(reward_batch[i] if terminal_state_batch[i]
                        #                     else reward_batch[i] + model.gamma * torch.max(next_state_batch_output[i])
                        #                           for i in range(len(minibatch)))
                        #
                        # # extract Q-value
                        # q_value = torch.sum(model(current_batch) * action_batch, dim=1)

                        # Use of Target Network
                        q_eval = torch.sum(model(current_batch) * action_batch, dim=1)  # shape (batch, 1)
                        q_next = target(next_state_batch).detach()     # detach from graph, don't backpropagate
                        q_target = reward_batch + 0.9 * q_next.max(1)[0]   # shape (batch, 1)
                        loss = criterion(q_eval, q_target)



                        # PyTorch accumulates gradients by default, so they need to be reset in each pass
                        optimizer.zero_grad()

                        # No use of Target Network
                        # returns a new Tensor, detached from the current graph, the result will never require gradient
                        # y_batch = torch.Tensor(y_batch).detach()

                        # calculate loss
                        # loss = criterion(q_value, y_batch)

                        if state % 10 == 0:
                            print('Epoch: {}, Runs: {}, Loss:   {:.4f}, Average Reward: {:.2f}'.format(epoch,index,loss.item(),sum(reward_batch)/self.minibatch_size))

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_wrapper = Data()
    data = data_wrapper.get_data()

    agent = Agent()

    featuresTrain = data_wrapper.get_training_data_tensor()

    predictor = RandomForestPredictor(data_wrapper.get_RFC_dataset())

    #TRAIN RL
    model = DeepQLearning()
    target_network = DeepQLearning()

    model.train(model,target_network,featuresTrain,agent)

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save("rl_model.pt")

if __name__ == '__main__':
    main()
