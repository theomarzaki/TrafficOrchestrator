#   Double Q-Learning implementation for the Traffic Orchestrator
#
#   This Script provides a way to approximate the best actions for the agent to undertake in order lane merge
#   Contains Random Forest Classifier to assign rewards to agent.
#
#       Created by Johan Maurel <johan.maurel@orange.com> (Orange Labs) for the 5GCar project
#

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


parser = argparse.ArgumentParser(description="TO RL : Double Q-Learning trainer")

parser.add_argument("-e", "--num-epochs", type=int, default=15, help="number of epochs to train (default: 10000)")
parser.add_argument("-o", "--learn-step-counter", type=int, default=0)
parser.add_argument("-n", "--number-of-actions", type=int, default=5)
parser.add_argument("-i", "--number-of-inputs", type=int, default=20)
parser.add_argument("-m", "--replay-memory-size", type=int, default=100000)
parser.add_argument("-d", "--epsilon-decay", type=int, default=100000)
parser.add_argument("-b", "--minibatch-size", type=int, default=32)
parser.add_argument("-z", "--hidden-layer-size", type=int, default=128)

parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
parser.add_argument("-g", "--gamma", type=float, default=0.9)
parser.add_argument("-f", "--final-epsilon", type=float, default=0.01)
parser.add_argument("-y", "--initial-epsilon", type=float, default=1.0)

args = parser.parse_args()


dev_type = 'cpu'

# if torch.cuda.is_available():
#     dev_type = 'cuda'
#     print("NVIDIA GPU detected and use !")
# else:
#     print("/!\ No NVIDIA GPU detected, we stick to the CPU !")

device = torch.device(dev_type)


class DoubleQLearning(nn.Module):
    def __init__(self, number_of_inputs, hidden_layer_size, number_of_actions):
        super(DoubleQLearning, self).__init__()
        self.number_of_actions = 5
        self.final_epsilon = 0.0001
        self.EPSILON_DECAY = 1000
        self.initial_epsilon = 1.0
        self.num_epochs = 25
        self.replay_memory_size = 1000000
        self.minibatch_size = 128
        self.gamma = 1
        self.learn_step_counter = 0
        self.learning_rate = 1e-3
        self.learning_rate_decay = 0.1


        self.layers = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )


    def forward(self, x):
        return self.layers(x)


    def train_double_dqn(self,model, target, featuresTrain, agent, predictor):
        replay_memory = []
        hist = []
        rewards = []
        wins = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        epsilon = model.initial_epsilon
        loss = 0

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

            sample_train = featuresTrain
            for index,game_run in enumerate(sample_train):
                for current_epoch,state in enumerate(game_run):
                    current = state
                    next = (sample_train[index])[current_epoch]

                    if self.learn_step_counter % 100 == 0:
                        target.load_state_dict(model.state_dict())
                    self.learn_step_counter += 1

                    output = model(current.to(device)).to(device)
                    # initialise actions

                    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
                    random_action = random.random() <= epsilon
                    action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                                    if random_action
                                    else torch.argmax(output)][0]

                    action[action_index] = 1

                    # get next state and reward

                    next_state = agent.calculateActionComputed(action,current,next)

                    reward,terminal = CalculateReward(next_state.data.cpu().numpy(),predictor)

                    rewards.append(reward)

                    # if replay memory is full, remove the oldest transition
                    if len(replay_memory) > model.replay_memory_size:
                        replay_memory.pop(0)


                    replay_memory.append((torch.Tensor(current).to(device), torch.Tensor(action).to(device), reward, torch.Tensor(next_state).to(device), terminal))


                    epsilon = model.learning_rate + (model.initial_epsilon - model.final_epsilon) * \
                                     math.exp(-1. * self.learn_step_counter / model.EPSILON_DECAY)


                    for g in optimizer.param_groups:
                        g['lr'] = model.final_epsilon + (1e-7 - 1e-3) * \
                                         math.exp(-1. * self.learn_step_counter / model.learning_rate_decay)


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

                    # # Use of Target Network
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

                    if(current_epoch % 70 == 0):
                        print('Epoch: {}/{},Runs: {}/{}, Loss: {:.6f}, Average Reward: {:.2f}, Epsilon: {:.4f}, Wins: {}'.format(epoch,self.num_epochs,index,featuresTrain.shape[0],loss.item(),sum(reward_batch)/self.minibatch_size,epsilon,wins))

                    # do backward pass
                    loss.backward()
                    optimizer.step()

                    if terminal == True:
                        wins = wins + 1
                        break
                    else:
                        try:
                            game_state[current_epoch + 1] = next_state
                        except:
                            # print("no more states (time) for maneuvers")
                            break

        return hist,rewards



def main():

    data_wrapper = Data()

    agent = Agent()

    featuresTrain = data_wrapper.get_training_data_tensor()

    predictor = RandomForestPredictor(data_wrapper.get_RFC_dataset())

    model = DoubleQLearning(args.number_of_inputs, args.hidden_layer_size, args.number_of_actions).to(device)
    local_network = DoubleQLearning(args.number_of_inputs, args.hidden_layer_size, args.number_of_actions).to(device)

    # state = torch.load('DQN_Saves/DQN6.tar', map_location='cpu')
    # model.load_state_dict(state['state_dict'])

    loss_over_time,rewards_over_time = model.train_double_dqn(model, local_network, featuresTrain, agent, predictor)

    plt.scatter(list(range(model.learn_step_counter)),loss_over_time)
    plt.show()

    plt.scatter(list(range(model.learn_step_counter)),rewards_over_time)
    plt.show()

    np.savetxt('double_loss.csv',loss_over_time)
    np.savetxt('double_rewards.csv',rewards_over_time,delimiter=',')

    traced_script_module = torch.jit.trace(model, torch.rand(args.number_of_inputs))
    traced_script_module.save("rl_model_double.pt")


if __name__ == '__main__':
    main()
