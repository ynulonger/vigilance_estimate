#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@author: Yilong Yang
#@time: 2018/03/21 15:58
########################################################
import scipy.io as sio
from scipy import stats
import numpy as np
import pandas as pd
import time
import math
import tensorflow as tf
import sys
import sklearn.preprocessing as preprocessing


CROSS_FOLDS = 5
data_dir = "./EEG_Feature_2Hz/"
label_dir = "./perclos_labels/"

input_height = 17
input_width = 25

def read_dataset(data_dir,label_dir,filename):
    final_dataset = np.empty([0,2,input_height,input_width])
    data = sio.loadmat(data_dir + filename)
    labels = sio.loadmat(label_dir + filename)["perclos"]
    dataset_de = np.transpose(data["de_movingAve"], [1, 0, 2])
    dataset_psd = np.transpose(data["psd_movingAve"], [1, 0, 2])
    for i in range(len(dataset_de)):
        temp = np.empty([0,input_height,input_width])
        data_de = preprocessing.scale(dataset_de[i].reshape(-1)).reshape(1,input_height,input_width)
        data_psd = preprocessing.scale(dataset_psd[i].reshape(-1)).reshape(1,input_height,input_width)
        temp = np.vstack([temp,data_psd])
        temp = np.vstack([temp,data_de])
        print("temp shape:",temp.shape)
        temp = temp.reshape(1,2,input_height,input_width)
        final_dataset = np.vstack([final_dataset,temp])
    return final_dataset,labels
data,labels = read_dataset(data_dir,label_dir,"chenting_20151124_noon_2.mat")
print(data.shape)