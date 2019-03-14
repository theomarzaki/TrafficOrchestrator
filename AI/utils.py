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
    try:
        if(round(x_diff,2) == 0 or round(y_diff,2) == 0 or slope == float('inf') or slope == float('-inf')):
            plus_c = int(state[8])
            if ((round(state[1]) + 1 == round(plus_c) or round(state[1]) - 1 == round(plus_c))):
                 if(int(state[7]) > int(state[0]) and int(state[0]) > int(state[13]) and int(state[8]) < int(state[1]) and int(state[1]) < int(state[14])):
                     return True;
        else:
            slope = round(y_diff,2) / round(x_diff,2)
            plus_c = state[8] - (slope * state[7])
            if(round(int(state[1])) in range(round(slope * int(state[0]) + plus_c) - 1, round(slope * int(state[0]) + plus_c) + 1)):
                if(int(state[7]) > int(state[0]) and int(state[0]) > int(state[13]) and int(state[8]) < int(state[1]) and int(state[1]) < int(state[14])):
                    return True; # C is on the line.
    except:
        pass
    return False;


def CalculateReward(state,predictor):
    # if predictor.predict_possible_merge(state[:19]) == False:
    #     reward = -1,False
    if isCarTerminal(state) == True:
        reward = 10000,True
    else:
        halfway = ((state[7] + state[13])/2,(state[8] + state[14])/2)
        distance_merging_point = distance.euclidean((state[0],state[1]), halfway)
        print(distance_merging_point)
        if(round(distance_merging_point,2) != 0):
            reward = (1/distance_merging_point),False
        else:
            reward = 0,False
    return reward

# def calculateDistance(pointA,pointB):
#     EARTH_RADIUS_KM = 6371.0
#     lat1r = math.radians(pointA[0]);
#     lon1r = math.radians(pointA[1]);
#     lat2r = math.radians(pointB[0]);
#     lon2r = math.radians(pointB[1]);
#     u = math.sin((lat2r - lat1r)/2);
#     v = math.sin((lon2r - lon1r)/2);
#     return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(u * u + math.cos(lat1r) * math.cos(lat2r) * v * v));


# ['globalXmerging', 'globalYmerging', 'lenghtMerging',
#        'widthMerging', 'velocityMerging', 'accelarationMerging',
#        'spacingMerging', 'globalXPreceding', 'globalYPreceding',
#        'lengthPreceding', 'widthPreceding', 'velocityPreceding',
#        'accelarationPreceding', 'globalXfollowing', 'globalYfollowing',
#        'widthFollowing', 'velocityFollowing', 'accelerationFollowing',
#        'spacingFollowing',] 'recommendation', 'heading',
#        'recommendedAcceleration'],
