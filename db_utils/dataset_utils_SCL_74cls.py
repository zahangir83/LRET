#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:00:36 2023

@author: malom
"""


import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import cv2
from os.path import join as join_path
import pdb
from collections import defaultdict
#from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage
import tensorflow as tf
import json
from json import JSONEncoder
from sklearn import preprocessing
from random import shuffle


kernel = np.ones((6,6), np.uint8) 

allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif']

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def color_perturbation(image):
      
      image = tf.image.random_brightness(image, max_delta=64./ 255.)
      image = tf.image.random_saturation(image, lower=0, upper=0.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, lower=0, upper=0.75)      
      
      return image

def flipRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def labels_encoder(acc_labels):
    encoded_obj = preprocessing.LabelEncoder()
    encoded_obj.fit(acc_labels)
    labels_encoded = encoded_obj.transform(acc_labels)
    return encoded_obj, labels_encoded
    
def labels_decoder(decoder_object,model_pred):
    
    decoded_pred = decoder_object.inverse_transform(model_pred)
    
    return decoded_pred

def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess_input_norm(x0):
    x = x0 / 255.
    return x
    

def samples_normalization (x_data, y_data):
    x_data = x_data.astype('float32')
    mean = np.mean(x_data)  # mean for data centering
    std = np.std(x_data)  # std for data normalization
    x_data -= mean
    x_data /= std
    
    y_data = y_data.astype('float32')
    y_data /= 255.  # scale masks to [0, 1]
    return x_data,y_data,mean,std


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
    
def shuffling_samples(data_x,data_y,data_y_encoded):    
    ## data_x and data_y are numpy array    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_y_encoded = np.array(data_y_encoded)

    perm = np.random.permutation(len(data_y))
    X = data_x[perm]
    y = data_y[perm]
    y_encoded = data_y_encoded[perm]
    
    return X,y,y_encoded
    
    
def unique_finder(list1):       
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set))    
    unique_labels = []
    for x in unique_list:        
        unique_labels.append(x)
        print(x)
       
    unique_labels = np.array(unique_labels)
    return unique_labels
        
def classe_logs(labels):
    
    classes_name = unique_finder(labels)
    num_classes = len(np.unique(labels))
    return classes_name, num_classes




def prepared_train_validation_DLF74cls_dataset_wrt_classes(idat, x_data, y_data, y_encoded_data, train_test_saving_path):
        
    end_labels = np.unique(y_encoded_data)
    
    train_idat_chip_id = []
    train_beta = []
    train_y_ac = []
    train_y_encoded = []
  
    val_idat_chip_id = []
    val_beta = []
    val_y_ac = []
    val_y_encoded = []
     
    
    
    for k in range(len(end_labels)):      
        idv_label = end_labels[k]   
        idx_for_idv_class = np.where(y_encoded_data == idv_label)
        
        print('The class is '+str(idv_label)+' and found samples: '+str(np.array(idx_for_idv_class).shape[1]))
        
        idat_for_idv_class = idat[idx_for_idv_class]
        beta_for_idv_class = x_data[idx_for_idv_class]
        y_data_for_idv_class = y_data[idx_for_idv_class]
        y_encoded_data_for_idv_class = y_encoded_data[idx_for_idv_class]
        
        idv_N = y_data_for_idv_class.shape[0] 
        ind_list = [i for i in range(idv_N)]
  
        shuffle(ind_list)    
        
        idv_idat_chip_id= idat_for_idv_class[ind_list]
        idv_beta = beta_for_idv_class[ind_list,:]
        idv_y_ac_labels = y_data_for_idv_class[ind_list]
        idv_y_encoded_labels = y_encoded_data_for_idv_class[ind_list]
        
        N = idv_y_ac_labels.shape[0] 
        train_end = int(N*0.80)
        val_test_samples = N - train_end 
        val_samples = int(val_test_samples*0.5)
        
        # training samples.....  
        idv_idat_chip_id_train = idv_idat_chip_id[0:train_end]
        idv_x_train = idv_beta[0:train_end,:]
        idv_y_ac_train = idv_y_ac_labels[0:train_end]
        idv_y_encoded_train = idv_y_encoded_labels[0:train_end]

        # .. Validation samples......
        idv_idat_chip_id_val = idv_idat_chip_id[train_end:]
        idv_x_val = idv_beta[train_end:,:]
        idv_y_ac_val = idv_y_ac_labels[train_end:]
        idv_y_encoded_val= idv_y_encoded_labels[train_end:]
        
        # # testing sampels....
        # test_start = train_end
        # idv_idat_chip_id_test = idv_idat_chip_id[test_start:]
        # idv_x_test = idv_beta[test_start:,:]
        # idv_y_ac_test = idv_y_ac_labels[test_start:]
        # idv_y_encoded_test = idv_y_encoded_labels[test_start:]
        
        #pdb.set_trace()
        
        if k == 0:         
            train_idat_chip_id = idv_idat_chip_id_train
            train_beta = idv_x_train
            train_y_ac = idv_y_ac_train
            train_y_encoded = idv_y_encoded_train
          
            val_idat_chip_id = idv_idat_chip_id_val
            val_beta = idv_x_val
            val_y_ac = idv_y_ac_val
            val_y_encoded = idv_y_encoded_val
             
            # test_idat_chip_id = idv_idat_chip_id_test
            # test_beta = idv_x_test
            # test_y_ac = idv_y_ac_test
            # test_y_encoded = idv_y_encoded_test
        else:
            
            train_idat_chip_id = np.concatenate((train_idat_chip_id, idv_idat_chip_id_train), axis = 0)
            train_beta = np.concatenate((train_beta, idv_x_train), axis = 0)
            train_y_ac = np.concatenate((train_y_ac, idv_y_ac_train), axis = 0)
            train_y_encoded = np.concatenate((train_y_encoded, idv_y_encoded_train), axis = 0)
            #val_idat_chip_id = idv_idat_chip_id_val
            #val_beta = idv_x_val
            #val_y_ac = idv_y_ac_val
            #val_y_encoded = idv_y_encoded_val   
            val_idat_chip_id = np.concatenate((val_idat_chip_id, idv_idat_chip_id_val), axis = 0)
            val_beta = np.concatenate((val_beta, idv_x_val), axis = 0)
            val_y_ac = np.concatenate((val_y_ac, idv_y_ac_val), axis = 0)
            val_y_encoded = np.concatenate((val_y_encoded, idv_y_encoded_val), axis = 0)
            # test_idat_chip_id = idv_idat_chip_id_test
            # test_beta = idv_x_test
            # test_y_ac = idv_y_ac_test
            # test_y_encoded = idv_y_encoded_test 
            # test_idat_chip_id = np.concatenate((test_idat_chip_id, idv_idat_chip_id_test), axis = 0)
            # test_beta = np.concatenate((test_beta, idv_x_test), axis = 0)
            # test_y_ac = np.concatenate((test_y_ac, idv_y_ac_test), axis = 0)
            # test_y_encoded = np.concatenate((test_y_encoded, idv_y_encoded_test), axis = 0)
        
        
        
        # train_idat_chip_id.append(idv_idat_chip_id_train)
        # train_beta.append(idv_x_train)
        # train_y_ac.append(idv_y_ac_train)
        # train_y_encoded.append(idv_y_encoded_train)
      
        # val_idat_chip_id.append(idv_idat_chip_id_val)
        # val_beta.append(idv_x_val)
        # val_y_ac.append(idv_y_ac_val)
        # val_y_encoded.append(idv_y_encoded_val)
         
        # test_idat_chip_id.append(idv_idat_chip_id_test)
        # test_beta.append(idv_x_test)
        # test_y_ac.append(idv_y_ac_test)
        # test_y_encoded.append(idv_y_encoded_test)
   
    #pdb.set_trace()
    
    train_idat_chip_id = np.array(train_idat_chip_id)
    train_beta = np.array(train_beta)
    train_y_encoded = np.array(train_y_encoded)
    train_y_ac = np.array(train_y_ac)
    
    N_train = train_y_encoded.shape[0] 
    ind_list_train = [i for i in range(N_train)]    
    shuffle(ind_list_train)  
    idat_chip_id_train= train_idat_chip_id[ind_list_train]     
    x_train = train_beta[ind_list_train,:]
    y_train = train_y_encoded[ind_list_train]
    y_ac_train = train_y_ac[ind_list_train]
    
    x_train_sp = join_path(train_test_saving_path,'x_train.npy')
    y_train_sp = join_path(train_test_saving_path,'y_train.npy')
    y_ac_train_sp = join_path(train_test_saving_path,'y_ac_train.npy')
    chip_id_train_sp = join_path(train_test_saving_path,'chip_id_train.npy')
    
    np.save(x_train_sp,x_train)
    np.save(y_train_sp,y_train)
    np.save(y_ac_train_sp,y_ac_train)
    np.save(chip_id_train_sp,idat_chip_id_train)
    
    # .. Validation samples......
    
    val_idat_chip_id = np.array(val_idat_chip_id)
    val_beta = np.array(val_beta)
    val_y_encoded = np.array(val_y_encoded)
    val_y_ac = np.array(val_y_ac)
    
    N_val = val_y_encoded.shape[0] 
    ind_list_val = [i for i in range(N_val)]    
    shuffle(ind_list_val)  
    idat_chip_id_val= val_idat_chip_id[ind_list_val]     
    x_val = val_beta[ind_list_val,:]
    y_val = val_y_encoded[ind_list_val]
    y_ac_val = val_y_ac[ind_list_val]
      
    x_val_sp = join_path(train_test_saving_path,'x_val.npy')
    y_val_sp = join_path(train_test_saving_path,'y_val.npy')
    y_ac_val_sp = join_path(train_test_saving_path,'y_ac_val.npy')
    chip_id_val_sp = join_path(train_test_saving_path,'chip_id_val.npy')
    
    np.save(x_val_sp,x_val)
    np.save(y_val_sp,y_val)
    np.save(y_ac_val_sp,y_ac_val)
    np.save(chip_id_val_sp,idat_chip_id_val)
     
     
    return x_train, y_train, y_ac_train, x_val, y_val, y_ac_val



