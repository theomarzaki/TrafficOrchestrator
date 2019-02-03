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


class Agent():
    def __init__(self):
        self.accelerate_tensor = 0
        self.deccelerate_tensor = 1
        self.left_tensor = 2
        self.right_tensor = 3
        self.doNothing_tensor = 4
        self.M_PI = 3.141539

    def calculateActionComputed(self,action_tensor,state):
        if (action_tensor == self.accelerate_tensor):
            return self.accelerate_move(state)
        elif (action_tensor == self.deccelerate_tensor):
            return self.deccelerate_move(state)
        elif (action_tensor == self.left_tensor):
            return self.left_move(state)
        elif (action_tensor == self.right_tensor):
            return self.right_move(state)
        elif (action_tensor == self.doNothing_tensor):
            return self.passive_move(state)
        else:
            logging.warning('inappropriate action -- SEE ME')

    def left_move(self,state):
        displacement = state[4] * 0.001 + 0.5 * (state[5] * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position - (0.001 * new_position)
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        return new_state

    def right_move(self,state):
        displacement = state[4] * 0.001 + 0.5 * (state[5] * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0]+ new_position + (0.001 * new_position)
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        return new_state

    def accelerate_move(self,state):
        final_velocity = state[4] + 0.001 * (state[4] + state[5] * 0.001)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.001)
        displacement = final_velocity * 0.001 + 0.5 * (final_acceleration * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        new_state[4] = final_velocity
        new_state[5] = final_acceleration
        return new_state

    def deccelerate_move(self,state):
        final_velocity = state[4] - 0.001 * (state[4] + state[5] * 0.001)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.001)
        displacement = final_velocity * 0.001 + 0.5 * (final_acceleration * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        new_state[4] = final_velocity
        new_state[5] = final_acceleration
        return new_state

    def passive_move(self,state):
        displacement = state[4] * 0.001 + 0.5 * (state[5] * 0.001 * 0.001)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + new_position
        new_y = state[1] + new_position
        new_state = state
        new_state[0] = new_x
        new_state[1] = new_y
        return new_state

model = torch.jit.load('rl_model.pt')

data = pd.read_csv("csv/lineMergeDataWithHeading.csv")
data.drop(['recommendation', 'heading', 'recommendedAcceleration'], axis=1, inplace=True)

featuresTrain = torch.zeros(math.ceil(data.shape[0]/70),70,19)

batch = torch.zeros(70,19)
counter = 0
for idx in range(data.shape[0]):
    if idx % 70 != 0 or idx == 0:
        batch[idx % 70]= torch.Tensor(data.values[idx])
    else:
        featuresTrain[counter] = batch
        counter = counter + 1
        batch = torch.zeros(70,19)

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
            except:
                pass
            output = model(torch.from_numpy(current))
            waypoint = agent.calculateActionComputed(torch.argmax(output),current)

            mergingX = waypoint[0]
            mergingY = waypoint[1]
            precedingX = next[7]
            precedingY = next[8]
            followingX = next[13]
            followingY = next[14]

            plots_X = [mergingX,precedingX,followingX]
            plots_Y = [mergingY,precedingY,followingY]
            to_plot.append((plots_X,plots_Y))


            try:
                game_state[state + 1] = torch.Tensor(next)
            except:
                break
        # all_plots.append(to_plot)
        # to_plot.clear()
        counter = counter + 1


camera = Camera(plt.figure())
colors = cm.rainbow(np.linspace(0, 1, 3))
for plot_data in to_plot:
    plt.scatter(plot_data[0],plot_data[1], c=colors, s=100)
    camera.snap()
anim = camera.animate(blit=True)
anim.save('trial.mp4')
