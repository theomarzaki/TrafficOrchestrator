# This Script provides extra methods that are used for simulating the enviroment

# Created by: Omar Nassef(KCL)

import pandas as pd
import numpy as np
import torch
import random
import math

def isCarTerminal(state):
    y_diff = state[14] - state[8]
    x_diff = state[13] - state[7]
    if(round(x_diff,2) == 0 or round(y_diff,2) == 0):
        plus_c = state[8]
        if ((round(state[1]) + 2 == round(plus_c) or round(state[1]) - 2 == round(plus_c))  and state[6] <= state[18]):
            return True;
    else:
        slope = round(y_diff,2) / round(x_diff,2)
        plus_c = state[8] - (slope * state[7])
        if ((round(state[1]) + 2 == round(slope * state[0] + plus_c) or round(state[1]) - 2 == round(slope * state[0]))  and state[6] <= state[18]):
            return True; # C is on the line.

    return False;


def CalculateReward(state,predictor):
    if predictor.predict_possible_merge(state[:19]) == False:
        reward = -1,True
    elif isCarTerminal(state) == True:
        reward = 1,True
    else:
        reward = -0.04,False

    return reward



# ['globalXmerging', 'globalYmerging', 'lenghtMerging',
#        'widthMerging', 'velocityMerging', 'accelarationMerging',
#        'spacingMerging', 'globalXPreceding', 'globalYPreceding',
#        'lengthPreceding', 'widthPreceding', 'velocityPreceding',
#        'accelarationPreceding', 'globalXfollowing', 'globalYfollowing',
#        'widthFollowing', 'velocityFollowing', 'accelerationFollowing',
#        'spacingFollowing',] 'recommendation', 'heading',
#        'recommendedAcceleration'],
