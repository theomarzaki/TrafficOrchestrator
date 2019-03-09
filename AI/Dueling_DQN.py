# This Script provides a way to approximate the best actions for the agent to undertake
# in order lane merge using Dueling DQN

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
from utils import isCarTerminal,CalculateReward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dueling_DQN(nn.Module):
    def __init__(self):
        super(Dueling_DQN,self).__init__()
        self.number_of_actions = 5
        self.final_epsilon = 0.01
        self.EPSILON_DECAY = 10000
        self.initial_epsilon = 1.0
        self.num_epochs = 50
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        self.gamma = 0.9
        self.learn_step_counter = 0
        self.learning_rate = 1e-4

        self.feature = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.number_of_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage  - advantage.mean()

    def update_target(current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def train_dueling(self,model,target,featuresTrain,agent,predictor):
        replay_memory = []
        hist = []
        wins = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        epsilon = model.initial_epsilon
        loss = None

        for epoch in range(self.num_epochs):

            model_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss':loss,
                }
            model_save_name = F"DQN{epoch}.tar"
            path = F"DQN_Saves/{model_save_name}"
            torch.save(model_state, path)

            for index,game_run in enumerate(featuresTrain):
                game_state = game_run
                # counter = counter + 1
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

                    if terminal and reward == 1:
                        wins = wins + 1
                    else:
                        pass

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
                    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
                    y_batch = tuple(reward_batch[i] if terminal_state_batch[i]
                                        else reward_batch[i] + model.gamma * torch.max(next_state_batch_output[i])
                                              for i in range(len(minibatch)))

                    # extract Q-value
                    q_value = torch.sum(model(current_batch) * action_batch, dim=1)

                    # Use of Target Network
                    # q_eval = torch.sum(model(current_batch).to(device) * action_batch, dim=1)  # shape (batch, 1)
                    # q_next = target(next_state_batch).to(device).detach()     # detach from graph, don't backpropagate
                    # q_target = reward_batch + 0.9 * q_next.max(1)[0]   # shape (batch, 1)
                    # loss = criterion(q_eval, q_target)



                    # PyTorch accumulates gradients by default, so they need to be reset in each pass
                    optimizer.zero_grad()

                    # No use of Target Network
                    # returns a new Tensor, detached from the current graph, the result will never require gradient
                    y_batch = torch.Tensor(y_batch).detach().to(device)

                    # calculate loss
                    loss = criterion(q_value, y_batch)

                    hist.append(loss.item())

                    if(state % 70 == 0):
                        print('Epoch: {}/{},Runs: {}/{}, Loss: {:.4f}, Average Reward: {:.2f}, Wins: {}'.format(epoch,self.num_epochs,index,featuresTrain.shape[0],loss.item(),sum(reward_batch)/self.minibatch_size,wins))

                    # do backward pass
                    loss.backward()
                    optimizer.step()

                    if terminal == True:
                        break
                    else:
                        try:
                            game_state[state + 1] = torch.Tensor(next_state).to(device)
                        except:
                            print("no more states (time) for maneuvers")
                            break

        return hist

def main():

    data_wrapper = Data()
    data = data_wrapper.get_data()


    agent = Agent()

    featuresTrain = data_wrapper.get_training_data_tensor()

    predictor = RandomForestPredictor(data_wrapper.get_RFC_dataset())


    model = Dueling_DQN().to(device)
    target_model = Dueling_DQN().to(device)


    #TRAIN RL
    # loss_over_time = model.train_dueling(model,target_model,featuresTrain,agent,predictor)
    # plt.plot(loss_over_time)
    # plt.show()

    # Load Model
    state = torch.load('DQN14.tar',map_location='cpu')
    model.load_state_dict(state['state_dict'])



    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save("rl_model_deuling.pt")


if __name__ == '__main__':
    main()
