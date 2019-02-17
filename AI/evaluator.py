# This script provides a way to test the implementation of the rl model in the merging scenario
# presented in the US101 dataset.
#
# output: a video file showing the agent(purple) in merge lane scenario

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


agent = Agent()
model = torch.jit.load('rl_model.pt')

data = Data().get_data()

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

to_plot = []
all_plots = []

agent = Agent()
counter = 0
for index,game_run in enumerate(featuresTrain):
    for current_epoch in range(game_run.shape[0]):
        game_state = game_run
        if counter > 3: break
        counter = counter + 1
        for state in range(game_state.shape[0]):
            current = game_state[state].data.cpu().numpy()
            try:
                next = game_state[state + 1].data.cpu().numpy()
                output = model(torch.from_numpy(current))
                waypoint = agent.calculateActionComputed(torch.argmax(output),current,next)
                # waypoint = agent.calculateActionComputed(0,current,next)


                mergingX = waypoint[0]
                mergingY = waypoint[1]
                precedingX = waypoint[7]
                precedingY = waypoint[8]
                followingX = waypoint[13]
                followingY = waypoint[14]

                plots_X = [mergingX,precedingX,followingX]
                plots_Y = [mergingY,precedingY,followingY]
                to_plot.append((plots_X,plots_Y))


                game_state[state + 1] = torch.Tensor(waypoint)
            except:
                pass
        counter = counter + 1


camera = Camera(plt.figure())
colors = cm.rainbow(np.linspace(0, 1, 3))
for plot_data in to_plot:
    plt.scatter(plot_data[0],plot_data[1], c=colors, s=100)
    camera.snap()
anim = camera.animate(blit=True)
anim.save('trial.mp4')
