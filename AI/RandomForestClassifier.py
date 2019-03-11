# This script is uses a random forest classifier to predict whether a car can merge or not

# @parameters output = model that can be used in python to train RL

# TODO :
# implement in pytorch to transfer to c++

# Created by: KCL
# Modified by: Omar Nassef(KCL)



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
import numpy as np

class RandomForestPredictor():
    def __init__(self,data):
        self.data = data
        self.train_and_test_random_forest_classifier()

    def countClassifiedResults(self,predictions):
        cnt = 0
        for i in range(len(predictions)):
            if predictions[i] != 1:
                cnt += 1

        return cnt, len(predictions)-cnt

    def train_and_test_random_forest_classifier(self):
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=1)
        self.train, self.validation = train_test_split(self.train_data, test_size=0.2, random_state=1)

        self.X_trainRecommendation = self.train.drop(['recommendation', 'heading', 'recommendedAcceleration','spacingMerging','spacingFollowing'], axis=1)
        self.Y_trainRecommendation = self.train['recommendation']

        self.X_valRecommendation = self.validation.drop(["recommendation", 'heading', 'recommendedAcceleration','spacingMerging','spacingFollowing'], axis=1).copy()
        self.Y_valRecommendation = self.validation['recommendation']

        self.X_testRecommendation = self.test_data.drop(['recommendation', 'heading', 'recommendedAcceleration','spacingMerging','spacingFollowing'], axis=1)
        self.Y_testRecommendation = self.test_data['recommendation']

        self.random_forest = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1)

        self.random_forest.fit(self.X_trainRecommendation, self.Y_trainRecommendation)

        acc_random_forest = round(self.random_forest.score(self.X_trainRecommendation, self.Y_trainRecommendation) * 100, 2)
        print("Training accuracy: ", acc_random_forest)

        self.Y_pred = self.random_forest.predict(self.X_valRecommendation)

        print(self.countClassifiedResults(self.Y_pred))

        acc_random_forest_val = round(accuracy_score(self.Y_valRecommendation, self.Y_pred)*100, 2)
        print("Validation accuracy: ", acc_random_forest_val)

    def predict_possible_merge(self,prediction_variables):
        pred = []
        for idx,data in enumerate(prediction_variables):
            if idx == 6 or idx == 18:
                pass
            else:
                pred.append(data)
        prediction_array = self.random_forest.predict(np.array(pred).reshape(1,-1))
        if len(prediction_array) != 0:
            if prediction_array[0] == 1:
                return True
            else:
                return False
        else:
            logging.warning('Trying to find predictions in an empty array')
