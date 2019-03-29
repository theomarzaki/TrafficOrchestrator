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
    plt.scatter(loss_over_time)
    plt.show()

    plt.scatter(rewards_over_time)
    plt.show()

    np.savetxt(F'{type}_loss.csv',loss_over_time)
    np.savetxt(F'{type}_rewards.csv',rewards_over_time,delimiter=',')

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save("rl_model_deuling.pt")

def load_checkpoint(model):
    state = torch.load('DQN_Saves/DQN2.tar',map_location='cpu')
    model.load_state_dict(state['state_dict'])

    traced_script_module = torch.jit.trace(model, torch.rand(20))
    traced_script_module.save("rl_model_deuling.pt")

def main():
    parser = argparse.ArgumentParser(description="TO RL : Double Q-Learning trainer")

    parser.add_argument("--dqn", action='store_true')
    parser.add_argument("--dueling", action='store_true')
    parser.add_argument("--double", action='store_true')

    args = parser.parse_args()

    if args.dqn:
        train(DQN().to(device),DQN().to(device),"DQN")
    elif args.dueling:
        train(Dueling_DQN().to(device),Dueling_DQN().to(device),"Dueling_DQN")
    else:
        train(DoubleQLearning().to(device),DoubleQLearning().to(device),"Double_DQN")

if __name__== "__main__":
    main()
