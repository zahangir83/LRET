#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:02:45 2023

@author: malom
"""

import numpy as np
import os
import shutil

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

# model.....
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

#from utils import helpers as helpers
#from db_utils import dataset_utils as data_utils
#from db_utils import shuf_split_utils
#from utils import features_utils as futils

import pdb

#subdirs, dirs, files = os.walk('/database/plant-seedlings-classification/train/').__next__()
#m = len(files)
#print(m)


#train_dir = '/Users/malom/Desktop/zahangir/projects/computational_pathology/Large_scale_histopathology_analysis/database/plant-seedlings-classification/train/'
#images_dir = './database/plant-seedlings-classification/images_all/'


def copy_train_image_in_single_dir(train_dir,images_dir):
    
    counter = 0
    
    for subdir, dirs, files in os.walk(train_dir):
        #print(files)
        for file in files:
            full_path = os.path.join(subdir, file)
            shutil.copy(full_path, images_dir)
            counter = counter + 1
    print(counter)


# DATA GENERATOR WITH FILE NAME 
class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size,image_size,image_dir) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.image_size = image_size
    self.image_dir = image_dir
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread(self.image_dir + str(file_name)), self.image_size)
               for file_name in batch_x])/255.0, np.array(batch_y)

# DATA GENERATOR WITH FILE PATH 
class My_Custom_Data_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_path, labels, batch_size,image_size) :
    self.image_path = image_path
    self.labels = labels
    self.batch_size = batch_size
    self.image_size = image_size
    #self.image_dir = image_dir
    
  def __len__(self) :
    return (np.ceil(len(self.image_path) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_path[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread(str(file_path)), self.image_size)
               for file_path in batch_x])/255.0, np.array(batch_y)

def generate_data_logs_from_dir_path_and_file_names(images_dir,log_saving_path):
    
    filepaths = []
    labels_actual = []
    
    for path, subdirs, files in os.walk(images_dir):        
        for dir_name in subdirs:            
            sub_dir = os.path.join(path, dir_name+'/')           
            print('Sub-dir:', sub_dir)   
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.tif' or '.png']    
            #images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.tif']    
            for filename in images:                  
                image_path = os.path.join(sub_dir,filename)  
                filepaths.append(image_path)
                labels_actual.append(dir_name)
    
    filepaths = np.array(filepaths)
    labels_actual = np.array(labels_actual)

    print('Files:',len(filepaths))
    print('Labels actual: ', len(labels_actual))
    
    filepaths_shuffled,labels_actual_shuffled = shuffle(filepaths, labels_actual)
    
    # saving the filename array as .npy file
    np.save(os.path.join(log_saving_path,'filepaths.npy'), filepaths_shuffled)
    np.save(os.path.join(log_saving_path,'labels_actual.npy'), labels_actual_shuffled)

    return filepaths_shuffled, labels_actual_shuffled

def generate_data_logs_from_dir(images_dir,log_saving_path):
    
    filenames = []
    labels = [] #np.zeros((m, 1))
    labels_actual = []
    
    filenames_counter = 0
    labels_counter = -1
    
    for subdir, dirs, files in os.walk(images_dir):
        #print(files)
        
        img_ext = files[0].split('.')[1]                
        #pdb.set_trace()
        if img_ext =='jpg' or 'png' or 'tif':      
                        
            for file in files:
                filenames.append(file)
                #full_path = os.path.join(subdir, file)
                #labels[filenames_counter, 0] = labels_counter
                labels.append(labels_counter)
                filenames_counter = filenames_counter + 1
                labels_actual.append(subdir)
                
            labels_counter = labels_counter+1
        
    
    filenames = np.array(filenames)
    labels = np.array(labels)
    labels_actual = np.array(labels_actual)
    
    # One hot vector representation of labels
    y_labels_one_hot = to_categorical(labels)

    print('Files:',len(filenames))
    print('Labels: ', labels.shape)
    print('Labels actual: ', len(labels_actual))
    print('One hot encoding :', y_labels_one_hot.shape)
    
    
    filenames_shuffled,labels_shuffled, labels_actual_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, labels, labels_actual, y_labels_one_hot)
    
    # saving the filename array as .npy file
    np.save(os.path.join(log_saving_path,'filenames.npy'), filenames_shuffled)
    # saving the filename array as .npy file
    np.save(os.path.join(log_saving_path,'labels_encoded.npy'), labels_shuffled)
    # saving the filename array as .npy file
    np.save(os.path.join(log_saving_path,'labels_actual.npy'), labels_actual_shuffled)
    # saving the y_labels_one_hot array as a .npy file
    np.save(os.path.join(log_saving_path,'y_labels_one_hot.npy'), y_labels_one_hot_shuffled)

    return filenames_shuffled,labels_shuffled, labels_actual_shuffled, y_labels_one_hot_shuffled