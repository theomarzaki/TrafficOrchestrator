# This script provides a way to test the implementation of the rl model in the merging scenario
# presented in the US101 dataset.
#
# @parameters output: a video file showing the agent(purple) in merge lane scenario

# Created by: Omar Nassef(KCL)


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
import matplotlib.animation as animation
from celluloid import Camera
from matplotlib import cm
from csv_data import Data
from Agent import Agent
import argparse
from utils import isCarTerminal, CalculateReward
from RandomForestClassifier import RandomForestPredictor

def FullMergeLaneScenario(is_scatter,featuresTrain,model,agent,predictor):
    to_plot = []
    counter = 0
    for index,game_run in enumerate(featuresTrain):
        for current_epoch in range(game_run.shape[0]):
            game_state = game_run
            if counter > 1: break
            counter = counter + 1
            for state in range(game_state.shape[0]):
                current = game_state[state].data.cpu().numpy()
                try:
                    next = game_state[state + 1].data.cpu().numpy()
                    output = model(torch.from_numpy(current))
                    action_tensor = torch.zeros(5)
                    action_tensor[torch.argmax(output)] = 1
                    waypoint = agent.calculateActionComputed(action_tensor,current,next)
                    reward,terminal = CalculateReward(waypoint,predictor)
                    print(reward)

                    if not isCarTerminal(waypoint):
                        if is_scatter:
                            to_plot.append((waypoint[0],waypoint[1]))
                        else:
                            mergingX = waypoint[0]
                            mergingY = waypoint[1]
                            precedingX = waypoint[7]
                            precedingY = waypoint[8]
                            followingX = waypoint[13]
                            followingY = waypoint[14]

                            plots_X = [mergingX,precedingX,followingX]
                            plots_Y = [mergingY,precedingY,followingY]
                            to_plot.append((plots_X,plots_Y))
                    else:
                        print("reached goal")
                        break

                    game_state[state + 1] = torch.Tensor(waypoint)
                except:
                    pass
    return to_plot

def ActionedMergeLaneScenario(actions,featuresTrain,agent,predictor):
    to_plot = []

    left = torch.Tensor([0,0,1,0,0])
    right = torch.Tensor([0,0,0,1,0])
    accelerate = torch.Tensor([1,0,0,0,0])
    decelerate = torch.Tensor([0,1,0,0,0])
    nothing = torch.Tensor([0,0,0,0,1])

    for index,action in enumerate(actions):
        game_state = (featuresTrain[0])[index]
        next = (featuresTrain[0])[index+1]
        next = next.cpu().numpy()
        if action == "left":
            waypoint = agent.calculateActionComputed(left,game_state,next)
        elif action == "right":
            waypoint = agent.calculateActionComputed(right,game_state,next)
        elif action == "accelerate":
            waypoint = agent.calculateActionComputed(accelerate,game_state,next)
        elif action == "decelerate":
            waypoint = agent.calculateActionComputed(decelerate,game_state,next)
        else:
            waypoint = agent.calculateActionComputed(nothing ,game_state,next)

        reward,terminal = CalculateReward(waypoint,predictor)
        try:
            waypoint = waypoint.cpu().numpy()
        except:
            pass
        print(reward)
        if not isCarTerminal(waypoint):
            mergingX = waypoint[0]
            mergingY = waypoint[1]
            precedingX = waypoint[7]
            precedingY = waypoint[8]
            followingX = waypoint[13]
            followingY = waypoint[14]

            plots_X = [mergingX,precedingX,followingX]
            plots_Y = [mergingY,precedingY,followingY]
            to_plot.append((plots_X,plots_Y))
        else:
            print("reached goal")
            break

        if index < 70:
            (featuresTrain[0])[index+1] = torch.Tensor(waypoint)
        else:
            break
    return to_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatter","--s",help="display vehicle trajectory of the merging car in a scatter plot",action='store_true')
    parser.add_argument("--heatmap","--h",help="display vehicle trajectory of the merging car in a heat map",action='store_true')
    parser.add_argument("--lstm","--lm",help="test the accuracy of the lstm model",action='store_true')
    parser.add_argument("--dueling_dqn","--ddqn",help="test the accuracy of the dueling DQN model",action='store_true')
    parser.add_argument("--actions","--a",help="show the movement of the agent with plain actions, requires array of actions",action='store_true')
    args = parser.parse_args()

    predictor = RandomForestPredictor(Data().get_RFC_dataset())
    agent = Agent()
    if args.dueling_dqn:
        model = torch.jit.load('rl_model_deuling.pt')
    elif args.lstm:
        lstm_model = torch.jit.load('../include/lstm_model.pt')
    else:
        model = torch.jit.load('rl_model.pt')

    data_wrapper = Data()
    data = data_wrapper.get_RFC_dataset()
    data = data[::-1]
    data.heading = (data.heading + 180) % 360
    data.drop(['recommendation', 'recommendedAcceleration'],axis=1,inplace=True)

    featuresTrain = torch.zeros(math.ceil(data.shape[0]/70),70,20)

    batch = torch.zeros(70,20)
    counter = 0
    for idx in range(data.shape[0]):
        if idx % 70 != 0 or idx == 0:
            batch[idx % 70]= torch.Tensor(data.values[idx])
        else:
            featuresTrain[counter] = batch
            counter = counter + 1
            batch = torch.zeros(70,20)

    if args.scatter == True:
        to_plot = FullMergeLaneScenario(True,featuresTrain,model,agent)
        plt.scatter([x[0] for x in to_plot],[x[1] for x in to_plot])
        plt.show()
    elif args.heatmap == True:
        print("coming soon ...")
    elif args.lstm == True:
        score = 0
        total = 0
        input,target = data_wrapper.get_testing_lstm_data()
        data_set = list(zip(input,target))
        for (input_data,target) in data_set:
            output = lstm_model(input_data.unsqueeze(0))
            print(output)
            if torch.equal(output,target):
                score = score + 1
            total = total + 1
        print((score/total)*100)
    elif args.actions == True:
        actions = ["left","left","left","decelerate","left","left","left","left","left","left","left","left",\
            "left","left","left","left","left","left","left","left","left","left","left","left","left","left",\
            "left","left","left","left","left","left","left","left","left","left","nothing","nothing",\
            "nothing","nothing","left","left","accelerate","accelerate","accelerate","accelerate","accelerate","accelerate","accelerate","accelerate","accelerate","accelerate",\
            "deccelerate","deccelerate","deccelerate","deccelerate","accelerate","deccelerate","accelerate","accelerate","deccelerate","accelerate","accelerate","left","left","accelerate","accelerate"]

        to_plot = ActionedMergeLaneScenario(actions,featuresTrain,agent,predictor)
        camera = Camera(plt.figure())
        colors = cm.rainbow(np.linspace(0, 1, 3))
        for plot_data in to_plot:
            plt.scatter(plot_data[0],plot_data[1], c=colors, s=100)
            camera.snap()
        anim = camera.animate(blit=True)
        anim.save('trial.mp4')
    else:
        to_plot = FullMergeLaneScenario(False,featuresTrain,model,agent,predictor)
        camera = Camera(plt.figure())
        colors = cm.rainbow(np.linspace(0, 1, 3))
        for plot_data in to_plot:
            plt.scatter(plot_data[0],plot_data[1], c=colors, s=100)
            camera.snap()
        anim = camera.animate(blit=True)
        anim.save('trial.mp4')

if __name__ == '__main__':
    main()
