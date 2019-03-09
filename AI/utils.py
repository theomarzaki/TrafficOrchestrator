# This Script provides extra methods that are used for simulating the enviroment

# Created by: Omar Nassef(KCL)

import pandas as pd
import numpy as np
import torch
import random
import math
from scipy.spatial import distance



def isCarTerminal(state):
    state  = state.tolist()
    y_diff = state[14] - state[8]
    x_diff = state[13] - state[7]
    if(round(x_diff,2) == 0 or round(y_diff,2) == 0):
        plus_c = int(state[8])
        if ((round(state[1]) + 2 == round(plus_c) or round(state[1]) - 2 == round(plus_c)) and int(state[6]) <= int(state[18])):
             if(int(state[7]) > int(state[0]) and int(state[0]) > int(state[13]) and int(state[8]) < int(state[1]) and int(state[1]) < int(state[14])):
                 return True;
    else:
        slope = round(y_diff,2) / round(x_diff,2)
        plus_c = state[8] - (slope * state[7])
        if(round(int(state[1])) in range(round(slope * int(state[0]) + plus_c) - 2,round(slope * int(state[0]) + plus_c) + 2) and int(state[6]) <= int(state[18])):
            if(int(state[7]) > int(state[0]) and int(state[0]) > int(state[13]) and int(state[8]) < int(state[1]) and int(state[1]) < int(state[14])):
                return True; # C is on the line.
    return False;


def CalculateReward(state,predictor):
    if predictor.predict_possible_merge(state[:19]) == False:
        reward = -1,True
    elif isCarTerminal(state) == True:
        reward = 1,True
    else:
        halfway = ((state[7] + state[13])/2,(state[8] + state[14])/2)
        # quarterway = ((halfway[0] + state[7])/2,(halfway[1] + state[8]/2))
        distance_merging_point = distance.euclidean((state[0],state[1]), halfway)
        if(round(distance_merging_point,2) != 0):
            reward = ((1/distance_merging_point)),False
        else:
            reward = 0
    return reward



# ['globalXmerging', 'globalYmerging', 'lenghtMerging',
#        'widthMerging', 'velocityMerging', 'accelarationMerging',
#        'spacingMerging', 'globalXPreceding', 'globalYPreceding',
#        'lengthPreceding', 'widthPreceding', 'velocityPreceding',
#        'accelarationPreceding', 'globalXfollowing', 'globalYfollowing',
#        'widthFollowing', 'velocityFollowing', 'accelerationFollowing',
#        'spacingFollowing',] 'recommendation', 'heading',
#        'recommendedAcceleration'],
