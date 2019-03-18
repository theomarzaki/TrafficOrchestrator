#!/usr/bin/env python

#   Double DQN implementation for the Traffic Orchestrator
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

from Agent import Agent
from RandomForestClassifier import RandomForestPredictor
from csv_data import Data
from utils import CalculateReward,isCarTerminal


parser = argparse.ArgumentParser(description="TO RL : Double DQN trainer")

parser.add_argument("-e","--num-epochs", type=int, default=25, help="number of epochs to train (default: 10000)")
parser.add_argument("-o","--learn-step-counter", type=int, default=0)
parser.add_argument("-n","--number-of-actions", type=int, default=5)
parser.add_argument("-m","--replay-memory-size", type=int, default=128000)
parser.add_argument("-d","--epsilon-decay", type=int, default=100000)
parser.add_argument("-b","--minibatch-size", type=int, default=32) 

parser.add_argument("-l","--learning-rate", type=float, default=0.0001)
parser.add_argument("-g","--gamma", type=float, default=0.9)
parser.add_argument("-f","--final-epsilon", type=float, default=0.01)
parser.add_argument("-i","--initial-epsilon", type=float, default=1.0)

args = parser.parse_args()


dev_type = 'cpu'

if torch.cuda.is_available():
    dev_type = 'cuda'
    print("NVIDIA GPU detected and use !")
else:
    print("/!\ No NVIDIA GPU detected, we stick to the CPU !")

device = torch.device(dev_type)


class DeepQLearning(nn.Module):
    def __init__(self):
        super(DeepQLearning,self).__init__()

        self.fc1 = nn.Linear(20,256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, args.number_of_actions)
        self.number_of_actions = 5
        self.final_epsilon = 0.01
        self.EPSILON_DECAY = 100000
        self.initial_epsilon = 1.0
        self.num_epochs = 10
        self.replay_memory_size = 10000
        self.minibatch_size = 32
        self.gamma = 0.9
        self.learn_step_counter = 0
        self.learning_rate = 1e-4

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def train(model,target,featuresTrain,agent,predictor):
    current_time = time.time()
    replay_memory = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    epsilon = args.initial_epsilon
    counter = 0
    wins = 0
    for epoch in range(args.num_epochs):
        for index,game_run in enumerate(featuresTrain):
            game_state = game_run
            counter = counter + 1
            for state in range(game_state.shape[0]):
                current = game_state[state].data.cpu().numpy()
                try:
                    s_next = game_state[state + 1].data.cpu().numpy()
                except:
                    pass

                if args.learn_step_counter % 100 == 0:
                    target.load_state_dict(model.state_dict())
                args.learn_step_counter += 1

                output = model(torch.from_numpy(current).to(device)).to(device)
                # initialise actions

                action = torch.zeros([args.number_of_actions], dtype=torch.float32)
                random_action = random.random() <= epsilon
                action_index = [torch.randint(args.number_of_actions, torch.Size([]), dtype=torch.int)
                                if random_action
                                else torch.argmax(output)][0]

                action[action_index] = 1

                # get next state and reward

                next_state = agent.calculateActionComputed(action,current,s_next)

                reward,terminal = CalculateReward(next_state,predictor)

                # if replay memory is full, remove the oldest transition
                if len(replay_memory) > args.replay_memory_size:
                    replay_memory.pop(0)


                replay_memory.append((torch.Tensor(current).to(device), torch.Tensor(action).to(device), reward, torch.Tensor(next_state).to(device), terminal))


                epsilon = args.final_epsilon + (args.initial_epsilon - args.final_epsilon) * \
                                 math.exp(-1. * args.learn_step_counter / args.epsilon_decay)

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

                next_state_batch_output = torch.zeros(args.minibatch_size,5).to(device)
                for idx in range(next_state_batch.shape[0]):
                    next_state_batch_output[idx] = model(next_state_batch[idx]).to(device)[0]


                # No use of Target Network
                # # set y_j to r_j for terminal state, otherwise to r_j +args.gamma*max(Q)
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

                if(args.learn_step_counter % 500 == 0):
                    model_state = {
                        'epoch': counter,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': loss,
                    }
                    model_save_name = F"DQN{counter}.tar"
                    path = F"DQN_Saves/{model_save_name}"
                    torch.save(model_state, path)

                if(state % 70 == 0):
                    print('Epoch: {}/{},Runs: {}/{}, Loss: {:.4f}, Average Reward: {:.2f}, Wins: {}'.format(epoch,
                                                                                                  args.num_epochs,
                                                                                                  index,
                                                                                                  featuresTrain.shape[0],
                                                                                                  loss.item(),
                                                                                                  sum(reward_batch)/args.minibatch_size,wins))
                bufftime = time.time()
                delta_time = bufftime - current_time
                print("+ {:.4f} Secs".format(delta_time))
                print("Perf: {:.4f} Epoch/s\n".format(1/delta_time))
                current_time = bufftime

                # do backward pass
                loss.backward()
                optimizer.step()

                if terminal == True:
                    wins = wins + 1
                    break
                else:
                    try:
                        game_state[state + 1] = torch.Tensor(next_state)
                    except:
                        print("no more states (time) for maneuvers")
                        break


def main():

    data_wrapper = Data()
    # data = data_wrapper.get_data()
    #
    agent = Agent()
    #
    featuresTrain = data_wrapper.get_training_data_tensor()
    #
    predictor = RandomForestPredictor(data_wrapper.get_RFC_dataset())

    model = DeepQLearning().to(device)
    target_network = DeepQLearning().to(device)

    # state = torch.load('rl_classifier.tar',map_location='cpu')
    # model.load_state_dict(state['state_dict'])

    train(model,target_network,featuresTrain,agent,predictor)

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save("rl_model_double.pt")

if __name__ == '__main__':
    main()
