#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:33:17 2020

@author: malom
"""

import numpy as np


def shuffle_input_samples(data_x,data_y,data_y_encoded):
    ## data_x and data_y are numpy array    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_y_encoded = np.array(data_y_encoded)

    perm = np.random.permutation(len(data_y))
    X = data_x[perm]
    y = data_y[perm]
    y_encoded = data_y_encoded[perm]
    
    return X,y,y_encoded

def splting_samples(x_data,y_data,y_ac_data):
    sample_count = len(y_data)   
    train_size = int(sample_count * 4 // 5)    
    
    #ac_x_train = ac_x_data[:train_size]
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    y_ac_train = y_ac_data[:train_size]
    
    #ac_x_val = ac_x_data[train_size:]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    y_ac_val = y_ac_data[train_size:]
    
    return x_train,y_train,y_ac_train, x_val,y_val, y_ac_val
    
def split_data(x_data,y_data,y_ac_data):

    sample_count = len(y_data)   
    train_size = int(sample_count * 4 // 5)    
    
    #ac_x_train = ac_x_data[:train_size]
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    y_ac_train = y_ac_data[:train_size]
    
    #ac_x_val = ac_x_data[train_size:]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    y_ac_val = y_ac_data[train_size:]

    
    return x_train,y_train,y_ac_train, x_val,y_val, y_ac_val