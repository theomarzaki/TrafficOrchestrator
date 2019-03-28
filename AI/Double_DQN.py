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


parser = argparse.ArgumentParser(description="TO RL : Double Q-Learning trainer")

parser.add_argument("-e", "--num-epochs", type=int, default=25, help="number of epochs to train (default: 10000)")
parser.add_argument("-n", "--number-of-actions", type=int, default=5)
parser.add_argument("-i", "--number-of-inputs", type=int, default=20)
parser.add_argument("-m", "--replay-memory-size", type=int, default=100000)
parser.add_argument("-b", "--minibatch-size", type=int, default=32)
parser.add_argument("-z", "--hidden-layer-size", type=int, default=128)

parser.add_argument("-d", "--epsilon-decay", type=float, default=0.99999)
parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
parser.add_argument("-g", "--discount", type=float, default=0.9)
parser.add_argument("-f", "--minimum-epsilon", type=float, default=0.001)
parser.add_argument("-y", "--initial-epsilon", type=float, default=1.0)
parser.add_argument("-t", "--tau", type=float, default=0.01)


dev_type = 'cpu'

# if torch.cuda.is_available():
#     dev_type = 'cuda'
#     print("NVIDIA GPU detected and use !")
# else:
#     print("/!\ No NVIDIA GPU detected, we stick to the CPU !")

device = torch.device(dev_type)


class DoubleQLearning(nn.Module):
    def __init__(self, args=None):
        super(DoubleQLearning, self).__init__()

        self.num_epochs = 25
        self.number_of_actions = 5
        self.number_of_inputs = 20
        self.replay_memory_size = 100000
        self.minibatch_size = 32
        self.hidden_layer_size = 128
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.0001
        self.discount = 0.9
        self.minimum_epsilon = 0.001
        self.initial_epsilon = 1.0
        self.tau = 0.01

        if args is not None:
            self.num_epochs = args.num_epochs
            self.number_of_actions = args.number_of_actions
            self.number_of_inputs = args.number_of_inputs
            self.replay_memory_size = args.replay_memory_size
            self.minibatch_size = args.minibatch_size
            self.hidden_layer_size = args.hidden_layer_size
            self.epsilon_decay = args.epsilon_decay
            self.learning_rate = args.learning_rate
            self.discount = args.discount
            self.minimum_epsilon = args.minimum_epsilon
            self.initial_epsilon = args.initial_epsilon
            self.tau = args.tau

        self.fc1 = nn.Linear(self.number_of_inputs, self.hidden_layer_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.number_of_actions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def train_double_qn(model, local, features_train, agent, predictor):

    learn_step_counter = 0
    current_time = time.time()
    replay_memory = []
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    criterion = nn.MSELoss()
    epsilon = model.initial_epsilon
    counter = 0
    wins = 0

    target = copy.deepcopy(local) # local.target = reward_next + discount*target.argmax(local.next_step)

    print("Go !")

    for epoch in range(model.num_epochs):
        for index, game_run in enumerate(features_train):

            game_state = game_run
            counter += 1

            for state in range(game_state.shape[0]):

                current = game_state[state].data.cpu().numpy()
                try:
                    s_next = game_state[state + 1].data.cpu().numpy()
                except:
                    pass

                if learn_step_counter % 100 == 0:
                    local.load_state_dict(model.state_dict())
                learn_step_counter += 1

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


                epsilon = (epsilon - model.minimum_epsilon) * model.epsilon_decay + model.minimum_epsilon

                minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

                current_batch = torch.zeros(len(minibatch),model.number_of_inputs).to(device)
                action_batch = torch.zeros(len(minibatch),model.number_of_actions).to(device)
                reward_batch = torch.zeros(len(minibatch)).to(device)
                next_state_batch = torch.zeros(len(minibatch),model.number_of_inputs).to(device)
                terminal_state_batch = []
                for idx,data_point in enumerate(minibatch):
                    current_batch[idx] = data_point[0]
                    action_batch[idx] = data_point[1]
                    reward_batch[idx] = data_point[2]
                    next_state_batch[idx] = data_point[3]
                    terminal_state_batch.append(data_point[4])

                next_state_batch_output = torch.zeros(model.minibatch_size,model.number_of_actions).to(device)
                for idx in range(next_state_batch.shape[0]):
                    next_state_batch_output[idx] = model(next_state_batch[idx]).to(device)[0]


                q_eval = torch.sum(model(current_batch).to(device) * action_batch, dim=1)  # shape (batch, 1)
                q_next = local(next_state_batch).to(device).detach()     # detach from graph, don't backpropagate
                q_local = reward_batch + model.discount * q_next.max(1)[0]
                loss = criterion(q_eval, q_local)

                # PyTorch accumulates gradients by default, so they need to be reset in each pass
                optimizer.zero_grad()


                if(learn_step_counter % 500 == 0):
                    model_state = {
                        'epoch': counter,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': loss,
                    }
                    model_save_name = "DQN"+str(counter)+".tar"
                    path = "DQN_Saves/"+model_save_name
                    torch.save(model_state, path)

                if(state % 500 == 0):
                    print('Epoch: {}/{},Runs: {}/{}, Loss: {:.4f}, Average Reward: {:.2f}, Eps: {:.6f}, Wins: {}'.format(epoch,
                                                                                                            model.num_epochs,
                                                                                                            index,
                                                                                                            features_train.shape[0],
                                                                                                            loss.item(),
                                                                                                            sum(reward_batch)/model.minibatch_size,
                                                                                                            epsilon,
                                                                                                            wins))

                # do backward pass
                loss.backward()
                optimizer.step()

                if terminal == True:
                    wins += 1
                    break
                else:
                    try:
                        game_state[state + 1] = torch.Tensor(next_state)
                    except:
                        bufftime = time.time()
                        delta_t = bufftime - current_time
                        print("+ {} Secs\nPerf: {} Step/s\n".format(delta_t,1/delta_t))
                        current_time = bufftime
                        break



def main():
    args = parser.parse_args()

    data_wrapper = Data()

    agent = Agent()

    features_train = data_wrapper.get_training_data_tensor()

    predictor = RandomForestPredictor(data_wrapper.get_RFC_dataset())

    model = DoubleQLearning().to(device)
    local_network = DoubleQLearning().to(device)

    # state = torch.load('rl_classifier.tar', map_location='cpu')
    # model.load_state_dict(state['state_dict'])

    train_double_qn(model, local_network, features_train, agent, predictor)

    traced_script_module = torch.jit.trace(model, torch.rand(model.number_of_inputs))
    traced_script_module.save("rl_model_double.pt")


if __name__ == '__main__':
    main()
