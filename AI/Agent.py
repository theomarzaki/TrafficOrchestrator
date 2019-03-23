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
        self.TIME_VARIANCE = 0.035

    def calculateActionComputed(self,action_tensor,state,next):
        if torch.equal(action_tensor,self.accelerate_tensor):
            # print("accelerate")
            return self.accelerate_move(state,next)
        elif torch.equal(action_tensor,self.deccelerate_tensor):
            # print("deccelerate")
            return self.deccelerate_move(state,next)
        elif torch.equal(action_tensor,self.left_tensor):
            # print("left")
            return self.left_move(state,next)
        elif torch.equal(action_tensor,self.right_tensor):
            # print("right")
            return self.right_move(state,next)
        elif torch.equal(action_tensor,self.doNothing_tensor):
            # print("nothing")
            return self.passive_move(state,next)
        else:
            logging.warning('inappropriate action')

    def left_move(self,state,next):
        displacement = state[4] * self.TIME_VARIANCE + 0.5 * (state[5] * self.TIME_VARIANCE * self.TIME_VARIANCE)
        angle = state[19]
        angle = (angle + 1) % 360
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = min(new_x,1e10)
        next[1] = min(new_y,1e10)
        next[19] = angle
        return next

    def right_move(self,state,next):
        displacement = state[4] * self.TIME_VARIANCE + 0.5 * (state[5] * self.TIME_VARIANCE * self.TIME_VARIANCE)
        angle = state[19]
        angle = (angle - 1) % 360
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = min(new_x,1e10)
        next[1] = min(new_y,1e10)
        next[19] = angle
        return next

    def accelerate_move(self,state,next):
        final_velocity = state[4] + self.TIME_VARIANCE * (state[4] + state[5] * self.TIME_VARIANCE)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * self.TIME_VARIANCE)
        displacement = final_velocity * self.TIME_VARIANCE + 0.5 * (final_acceleration * self.TIME_VARIANCE * self.TIME_VARIANCE)
        angle = state[19]
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = min(new_x,1e10)
        next[1] = min(new_y,1e10)
        next[4] = final_velocity
        next[5] = final_acceleration
        next[19] = angle
        return next

    def deccelerate_move(self,state,next):
        final_velocity = state[4] - self.TIME_VARIANCE * (state[4] + state[5] * self.TIME_VARIANCE)
        final_acceleration = (math.pow(final_velocity,2) - math.pow(state[4],2)) / 2 * (0.5 * (state[4] + final_velocity) * self.TIME_VARIANCE)
        displacement = final_velocity * self.TIME_VARIANCE + 0.5 * (final_acceleration * self.TIME_VARIANCE * self.TIME_VARIANCE)
        angle = state[19]
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = min(new_x,1e10)
        next[1] = min(new_y,1e10)
        next[4] = final_velocity
        next[5] = final_acceleration
        next[19] = angle
        return next

    def passive_move(self,state,next):
        displacement = state[4] * self.TIME_VARIANCE + 0.5 * (state[5] * self.TIME_VARIANCE * self.TIME_VARIANCE)
        angle = state[19]
        new_x = state[0] + displacement * math.cos(math.radians(angle))
        new_y = state[1] + displacement * math.sin(math.radians(angle))
        next[0] = min(new_x,1e10)
        next[1] = min(new_y,1e10)
        next[19] = state[19]
        return next
