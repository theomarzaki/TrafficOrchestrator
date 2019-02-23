# This script provides a way for the agent(car) to carry out actions based on its

#various values

# @parameters input: state tensor of highway

# @parameters output: new state tensor

# Created by: Omar Nassef(KCL)


import pandas as pd
import numpy as np
import torch
import math

class Agent():
    def __init__(self):
        self.accelerate_tensor = torch.Tensor([1,0,0,0,0])
        self.deccelerate_tensor = torch.Tensor([0,1,0,0,0])
        self.left_tensor = torch.Tensor([0,0,1,0,0])
        self.right_tensor = torch.Tensor([0,0,0,1,0])
        self.doNothing_tensor = torch.Tensor([0,0,0,0,1])

    def calculateActionComputed(self,action_tensor,state,next):
        if torch.equal(action_tensor,self.accelerate_tensor):
            return self.accelerate_move(state,next)
        elif torch.equal(action_tensor,self.deccelerate_tensor):
            return self.deccelerate_move(state,next)
        elif torch.equal(action_tensor,self.left_tensor):
            return self.left_move(state,next)
        elif torch.equal(action_tensor,self.right_tensor):
            return self.right_move(state,next)
        elif torch.equal(action_tensor,self.doNothing_tensor):
            return self.passive_move(state,next)
        else:
            logging.warning('inappropriate action')

    def left_move(self,state,next):
        displacement = state[4] * 0.035 + 0.5 * (state[5] * 0.035 * 0.035)
        angle = state[19]
        angle = (angle + 5) % 360
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = new_x
        next[1] = new_y
        next[19] = angle
        return next

    def right_move(self,state,next):
        displacement = state[4] * 0.035 + 0.5 * (state[5] * 0.035 * 0.035)
        angle = state[19]
        angle = (angle - 5) % 360
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = new_x
        next[1] = new_y
        next[19] = angle
        return next

    def accelerate_move(self,state,next):
        final_velocity = state[4] + 0.035 * (state[4] + state[5] * 0.035)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.035)
        displacement = final_velocity * 0.035 + 0.5 * (final_acceleration * 0.035 * 0.035)
        angle = state[19]
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = new_x
        next[1] = new_y
        next[4] = final_velocity
        next[5] = final_acceleration
        return next

    def deccelerate_move(self,state,next):
        final_velocity = state[4] - 0.035 * (state[4] + state[5] * 0.035)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * 0.035)
        displacement = final_velocity * 0.035 + 0.5 * (final_acceleration * 0.035 * 0.035)
        angle = state[19]
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = new_x
        next[1] = new_y
        next[4] = final_velocity
        next[5] = final_acceleration
        return next

    def passive_move(self,state,next):
        displacement = state[4] * 0.035 + 0.5 * (state[5] * 0.035 * 0.035)
        angle = state[19]
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = new_x
        next[1] = new_y
        return next
