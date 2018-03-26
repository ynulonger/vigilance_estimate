#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@file: pre_process.py
#@author: Yilong Yang
#@time: 2017/12/21 15:58
########################################################
import scipy.io as sio
import os
import numpy as np

np.random.seed(0)

data_dir = "./EEG_Feature_2Hz/"
label_dir = "./perclos_labels/"
file_list = os.listdir(data_dir)
for data_file in file_list:
    data = sio.loadmat(data_dir+data_file)
    label = sio.loadmat(label_dir+data_file)
    print(np.transpose(data["de_movingAve"],[1,0,2]).shape,"--",label["perclos"].shape)
