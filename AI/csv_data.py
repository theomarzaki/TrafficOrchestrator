# This script provides a way for the different models to access the training and testing data

# Created by: Omar Nassef(KCL)


import pandas as pd
import torch
import math
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

class Data():
    def __init__(self):
        self.data = pd.read_csv("csv/lineMergeDataWithHeading.csv")
        self.data = self.data[::-1]
        self.data.heading = (self.data.heading + 180) % 360
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2, random_state=1)
        self.scaler = MinMaxScaler(feature_range =(-1, 1))

    def get_data(self):
        return self.data

    def get_RFC_dataset(self):
        return pd.read_csv("csv/lineMergeDataWithHeading.csv")

    def get_training_data_tensor(self):
        self.featuresTrain = torch.zeros(math.ceil(self.train_data.shape[0]/70),70,20)
        self.train_data.drop(['recommendation', 'recommendedAcceleration'],axis=1,inplace=True)
        batch = torch.zeros(70,20)
        counter = 0
        for idx in range(self.train_data.shape[0]):
            if idx % 70 != 0 or idx == 0:
                batch[idx % 70]= torch.Tensor(self.train_data.values[idx])
            else:
                self.featuresTrain[counter] = batch
                counter = counter + 1
                batch = torch.zeros(70,20)

        return self.featuresTrain

    def get_testing_data_tensor(self):
        self.featuresTest = torch.zeros(math.ceil(self.test_data.shape[0]/70),70,20)
        self.test_data.drop(['recommendation', 'recommendedAcceleration'],axis=1,inplace=True)
        batch = torch.zeros(70,20)
        counter = 0
        for idx in range(self.test_data.shape[0]):
            if idx % 70 != 0 or idx == 0:
                batch[idx % 70]= torch.Tensor(self.test_data.values[idx])
            else:
                self.featuresTest[counter] = batch
                counter = counter + 1
                batch = torch.zeros(70,20)

        return self.featuresTest

    def get_training_lstm_data(self):
        self.train_data.drop(['recommendation', 'recommendedAcceleration','widthMerging','lenghtMerging','spacingMerging','lengthPreceding','widthPreceding','widthFollowing','spacingFollowing'], axis=1, inplace=True)
        scaled = self.scaler.fit_transform(self.train_data)
        # scaled = self.train_data
        featuresTrain = torch.zeros(math.ceil(scaled.shape[0]/2),1,13)
        targetsTrain = torch.zeros(math.ceil(scaled.shape[0]/2),1,13)
        f_counter = 0
        t_counter = 0
        for idx in range(scaled.shape[0]):
            if idx % 2 != 0:
                featuresTrain[f_counter] = torch.Tensor(scaled[idx])
                f_counter = f_counter + 1
            else:
                targetsTrain[t_counter] = torch.Tensor(scaled[idx])
                t_counter = t_counter + 1
        return featuresTrain,targetsTrain

    def get_testing_lstm_data(self):
        self.test_data.drop(['recommendation', 'recommendedAcceleration','widthMerging','lenghtMerging','spacingMerging','lengthPreceding','widthPreceding','widthFollowing','spacingFollowing'], axis=1, inplace=True)
        featuresTest = torch.zeros(math.ceil(self.train_data.shape[0]/2),1,13)
        targetsTest = torch.zeros(math.ceil(self.train_data.shape[0]/2),1,13)
        f_counter = 0
        t_counter = 0
        for idx in range(self.train_data.shape[0]):
            if idx % 2 != 0:
                featuresTest[f_counter] = torch.Tensor(self.train_data.values[idx])
                f_counter = f_counter + 1
            else:
                targetsTest[t_counter] = torch.Tensor(self.train_data.values[idx])
                t_counter = t_counter + 1
        return featuresTest,targetsTest