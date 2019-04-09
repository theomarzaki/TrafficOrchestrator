import torch
import torch.nn as nn
import math
import random
import argparse
import time
import copy

from Agent import Agent
from RandomForestClassifier import RandomForestPredictor
from csv_data import Data
from utils import CalculateReward
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUMBER_OF_ACTIONS = 5
NUMBER_OF_INPUTS = 20
FINAL_EPSILON = 0.01
EPSILON_DECAY = 500000
INITIAL_EPSILON = 1.0
REPLAY_MEMORY_SIZE = 1000000
NUMBER_OF_EPOCHS = 10
MINIBATCH_SIZE = 32
GAMMA = 0.9
LEARNING_RATE = 1e-2

def save_model_checkpoint(epoch,model_network,optimizer,loss):
    model_state = {
        'epoch': epoch,
        'state_dict': model_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss':loss,
        }

    model_save_name = F"DQN{epoch}.tar"
    path = F"DQN_Saves/{model_save_name}"
    torch.save(model_state, path)


def minibatch_reward(model,replay_memory):
    minibatch = random.sample(replay_memory, min(len(replay_memory), MINIBATCH_SIZE))

    current_batch = torch.zeros(len(minibatch),NUMBER_OF_INPUTS).to(device)
    action_batch = torch.zeros(len(minibatch),NUMBER_OF_ACTIONS).to(device)
    reward_batch = torch.zeros(len(minibatch)).to(device)
    next_state_batch = torch.zeros(len(minibatch),NUMBER_OF_INPUTS).to(device)
    terminal_state_batch = []
    for idx,data_point in enumerate(minibatch):
        current_batch[idx] = data_point[0]
        action_batch[idx] = data_point[1]
        reward_batch[idx] = data_point[2]
        next_state_batch[idx] = data_point[3]
        terminal_state_batch.append(data_point[4])

    next_state_batch_output = torch.zeros(MINIBATCH_SIZE,NUMBER_OF_INPUTS).to(device)
    for idx in range(next_state_batch.shape[0]):
        next_state_batch_output[idx] = model(next_state_batch[idx]).to(device)[0]

    y_batch = tuple(reward_batch[i] if terminal_state_batch[i]
                        else reward_batch[i] + GAMMA * torch.max(next_state_batch_output[i])
                              for i in range(len(minibatch)))

    return y_batch,current_batch,action_batch,reward_batch,next_state_batch

def train_model(model_network,target_network,train_data,agent):
    learn_step_counter = 0
    loss = 0
    wins = 0
    replay_memory = []
    hist = []
    rewards = []


    optimizer = torch.optim.Adam(model_network.parameters())
    criterion = nn.SmoothL1Loss()
    epsilon = INITIAL_EPSILON

    for epoch in range(NUMBER_OF_EPOCHS):

        save_model_checkpoint(epoch,model_network,optimizer,loss)

        sample_train = train_data
        for index,game_run in enumerate(sample_train):
            for current_epoch,state in enumerate(game_run):
                current = state
                if(current_epoch < game_run.shape[0] - 1):
                    next = game_run[current_epoch + 1]
                else:
                    break

                if learn_step_counter % 100 == 0:
                    target_network.load_state_dict(model_network.state_dict())
                learn_step_counter += 1

                output = model_network(current.to(device)).to(device)
                # initialise actions

                action = torch.zeros([NUMBER_OF_ACTIONS], dtype=torch.float32)
                random_action = random.random() < epsilon
                # action_index = [torch.randint(NUMBER_OF_ACTIONS, torch.Size([]), dtype=torch.int)
                #                 if random_action
                #                 else torch.argmax(output)][0]

                if random_action:
                    action[random.randrange(0,5)] = 1
                else:
                    action[torch.argmax(output).item()] = 1


                # get next state and reward

                next_state = agent.calculateActionComputed(action,current,next)

                reward,terminal = CalculateReward(next_state.data.cpu().numpy(),current_epoch)

                rewards.append((learn_step_counter,reward))
                # print(reward)
                if terminal and reward > 1:
                    wins = wins + 1

                if len(replay_memory) > REPLAY_MEMORY_SIZE:
                    replay_memory.pop(0)


                replay_memory.append((torch.Tensor(current).to(device), torch.Tensor(action).to(device), reward, torch.Tensor(next_state).to(device), terminal))


                epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * \
                                 math.exp(-1. * learn_step_counter / EPSILON_DECAY)


                y_batch,current_batch,action_batch,reward_batch,next_state_batch = minibatch_reward(model_network,replay_memory)

                # extract Q-value
                q_value = torch.sum(model_network(current_batch) * action_batch, dim=1)

                q_eval = torch.sum(model_network(current_batch).to(device) * action_batch, dim=1)  # shape (batch, 1)
                q_next = target_network(next_state_batch).to(device).detach()     # detach from graph, don't backpropagate
                q_target = reward_batch + 0.9 * q_next.max(1)[0] * 1 - int(terminal)  # shape (batch, 1)
                loss = criterion(q_eval, q_target)

                optimizer.zero_grad()

                # y_batch = torch.Tensor(y_batch).detach().to(device)
                # calculate loss
                # loss = criterion(q_value, y_batch)

                hist.append((learn_step_counter,loss.item()))
                # do backward pass
                loss.backward()
                optimizer.step()

                if terminal == True:
                    break
                if(current_epoch < game_run.shape[0] - 1):
                    game_run[current_epoch + 1] = next_state
                else:
                    break
            print('Epoch: {}/{},Runs: {}/{}, Loss: {:.6f}, Average Reward: {:.2f}, Epsilon: {:.4f}, Wins: {}'.format(epoch,NUMBER_OF_EPOCHS,index,train_data.shape[0],loss.item(),sum(reward_batch)/MINIBATCH_SIZE,epsilon,wins))
    return hist,rewards
