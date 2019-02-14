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

    def calculateActionComputed(self,action_tensor,state,next):
        if (action_tensor == self.accelerate_tensor):
            return self.accelerate_move(state,next)
        elif (action_tensor == self.deccelerate_tensor):
            return self.deccelerate_move(state,next)
        elif (action_tensor == self.left_tensor):
            return self.left_move(state,next)
        elif (action_tensor == self.right_tensor):
            return self.right_move(state,next)
        elif (action_tensor == self.doNothing_tensor):
            return self.passive_move(state,next)
        else:
            logging.warning('inappropriate action -- SEE ME')

    def left_move(self,state,next):
        displacement = state[4] * 0.01 + 0.5 * (state[5] * 0.01 * 0.01)
        angle = state[20]
        print(angle)
        # angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        # new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))

        if(angle <= 180):
            angle = (angle - 5) % 360
        else:
            angle = (angle + 5) % 360

        print(angle)
        new_x = state[0] + displacement * Math.Cos(angle * Math.Pi / 180)
        new_y = state[1] + displacement * Math.Sin(angle * Math.Pi / 180)
        next[0] = new_x
        next[1] = new_y
        return next

    def right_move(self,state,next):
        displacement = state[4] * 0.01 + 0.5 * (state[5] * 0.01 * 0.01)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        # new_x = state[0]+ new_position + (0.01 * new_position)
        # new_x = state[0] + (0.1 * new_position)
        new_y = state[1] - 1.1 * (new_position)
        # next[0] = new_x
        next[1] = new_y
        return next

    def accelerate_move(self,state,next):
        final_velocity = state[4] + 0.01 * (state[4] + state[5] * 0.01)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.01)
        displacement = final_velocity * 0.01 + 0.5 * (final_acceleration * 0.01 * 0.01)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] + 1.1 * new_position
        # new_y = state[1] + new_position
        next[0] = new_x
        # next[1] = new_y
        next[4] = final_velocity
        next[5] = final_acceleration
        return next

    def deccelerate_move(self,state,next):
        final_velocity = state[4] - 0.01 * (state[4] + state[5] * 0.01)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.01)
        displacement = final_velocity * 0.01 + 0.5 * (final_acceleration * 0.01 * 0.01)
        angular_displacement = math.degrees(math.sin(15)) * displacement / math.degrees(math.sin (90))
        new_position = math.sqrt(pow(angular_displacement,2) + pow(displacement,2))
        new_x = state[0] - 1.1 * new_position
        # new_y = state[1] + new_position
        next[0] = new_x
        # next[1] = new_y
        next[4] = final_velocity
        next[5] = final_acceleration
        return next

    def passive_move(self,state,next):
        return next

model = torch.jit.load('rl_model.pt')

data = pd.read_csv("csv/lineMergeDataWithHeading.csv")
data.drop(['recommendation', 'recommendedAcceleration'], axis=1, inplace=True)
data = data[::-1]
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
                # output = model(torch.from_numpy(current))
                # waypoint = agent.calculateActionComputed(torch.argmax(output),current)
                waypoint = agent.calculateActionComputed(2,current,next)

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
