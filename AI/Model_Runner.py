import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent
from csv_data import Data
from RL_Model import DQN,Dueling_DQN,DoubleQLearning
from trainer import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model,target_model,type):

    data_wrapper = Data()
    data = data_wrapper.get_data()
    agent = Agent()

    #TRAIN RL
    featuresTrain = data_wrapper.get_training_data_tensor()
    loss_over_time,rewards_over_time = train_model(model,target_model,featuresTrain,agent)

    try:
        plt.plot(loss_over_time)
        plt.show()

        plt.plot(rewards_over_time)
        plt.show()
    except:
        pass

    np.savetxt(F'{type}_loss.csv',loss_over_time)
    np.savetxt(F'{type}_rewards.csv',rewards_over_time,delimiter=',')

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save(F"rl_model_{type}.pt")

def load_checkpoint(model,model_name,save_number):
    state = torch.load(F'DQN_Saves/DQN{save_number}.tar',map_location='cpu')
    model.load_state_dict(state['state_dict'])

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save(F"rl_model_{model_name}.pt")

def train_checkpoint(model,model_name,save_number):
    state = torch.load(F'DQN_Saves/DQN{save_number}.tar',map_location='cpu')
    model.load_state_dict(state['state_dict'])
    target = model
    train(model.to(device),target.to(device),model_name)


def main():
    parser = argparse.ArgumentParser(description="TO RL Trainer")

    parser.add_argument("--dqn", action='store_true')
    parser.add_argument("--dueling", action='store_true')
    parser.add_argument("--double", action='store_true')
    parser.add_argument("--load","--l", action='store_true')
    parser.add_argument("--checkpoint_number","--cn", type=int, default = 3, help='loading checkpoint to train')
    parser.add_argument("--checkpoint_train","--ct", action='store_true')

    args = parser.parse_args()

    if args.dqn:
        model_network = target_network = DQN()
        model_name = "DQN"
    elif args.dueling:
        model_network = target_network = Dueling_DQN()
        model_name = "Dueling"
    else:
        model_network = target_network = DoubleQLearning()
        model_name = "Double"

    if args.load:
        load_checkpoint(model_network,model_name,args.checkpoint_number)
    elif args.checkpoint_train:
        train_checkpoint(model_network,model_name,args.checkpoint_number)
    else:
        train(model_network.to(device),target_network.to(device),model_name)


if __name__== "__main__":
    main()
