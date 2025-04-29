# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:47:54 2019
@author: deeplens
"""
"""PatchCamelyon(PCam) dataset
Small 96x96 patches from histopathology slides from the Camelyon16 dataset.
Please consider citing [1] when used in your publication:
- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". 
arXiv [cs.CV] (2018), (available at http://arxiv.org/abs/1806.03962).
Author: Bastiaan Veeling
Source: https://github.com/basveeling/pcam
"""
import os
import numpy as np
import pandas as pd
from keras.utils import HDF5Matrix
from keras.utils.io_utils import HDF5Matrix
from keras.utils.data_utils import get_file
from keras import backend as K

import pdb


def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x
    

def get_unzip_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    import gzip
    import shutil
    get_file()
    with open('file.txt', 'rb') as f_in, gzip.open('file.txt.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def pcam_database_loader_v1(dirname):
    """Loads PCam dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    
    #dirname = os.path.join('datasets', 'pcam')
    #base = 'https://drive.google.com/uc?export=download&id='
    try:
        x_train = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_train_x.h5'),'x')
        y_train = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_train_y.h5'),'y')
        x_valid = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_valid_x.h5'),'x')
        y_valid = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_valid_y.h5'),'y')
        x_test = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_test_x.h5'),'x')
        y_test = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_test_y.h5'),'y')

        meta_train = pd.read_csv(os.path.join(dirname,'camelyonpatch_level_2_split_train_meta.csv'))
        meta_valid = pd.read_csv(os.path.join(dirname,'camelyonpatch_level_2_split_valid_meta.csv'))
        meta_test = pd.read_csv(os.path.join(dirname,'camelyonpatch_level_2_split_test_meta.csv'))
       
    except OSError:
        raise NotImplementedError('Direct download currently not working.')
        
    if K.image_data_format() == 'channels_first':
        raise NotImplementedError()

    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)


def pcam_database_loader_training(dirname):
    """Loads PCam dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    
    # loading all of the samples....
    try:
        x_train = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_train_x.h5'),'x')
        y_train = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_train_y.h5'),'y')
        x_valid = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_valid_x.h5'),'x')
        y_valid = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_valid_y.h5'),'y')
       
    except OSError:
        raise NotImplementedError('Direct download currently not working.')
        
    if K.image_data_format() == 'channels_first':
        raise NotImplementedError()
    
    # preprocessing training samples...
    x_train = np.array(x_train).astype(np.float32)
    x_train=x_train.transpose((0,1,2,3))
    x_train = preprocess_input(x_train)    
    y_train = np.array(y_train)
    y_train = np.squeeze(y_train)
    perm_train = np.random.permutation(len(y_train))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]
    
   
    # preprocessing training samples...
    x_valid = np.array(x_valid ).astype(np.float32)
    x_valid =x_valid .transpose((0,1,2,3))
    x_valid  = preprocess_input(x_valid )    
    y_valid  = np.array(y_valid )
    y_valid  = np.squeeze(y_valid )
    perm_valid  = np.random.permutation(len(y_valid ))
    x_valid  = x_valid[perm_valid]
    y_valid  = y_valid[perm_valid]
    
    # tags are the unique values of the array.... 
    
    tags = np.unique(y_train)
    
    return (x_train, y_train), (x_valid, y_valid), tags

def pcam_database_loader_testing(dirname):
    """Loads PCam dataset.
    # Returns
        Tuple of Numpy arrays: (x_test, y_test)`.
    """
    
    # loading all of the samples....
    try:
        x_test = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_test_x.h5'),'x')
        y_test = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_test_y.h5'),'y')
        
        x_valid = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_valid_x.h5'),'x')
        y_valid = HDF5Matrix(os.path.join(dirname,'camelyonpatch_level_2_split_valid_y.h5'),'y')
       
    except OSError:
        raise NotImplementedError('Direct download currently not working.')
        
    if K.image_data_format() == 'channels_first':
        raise NotImplementedError()
    
    # preprocessing training samples...
    x_test = np.array(x_test).astype(np.float32)
    x_test=x_test.transpose((0,1,2,3))
    x_test = preprocess_input(x_test)    
    y_test = np.array(y_test)
    y_test = np.squeeze(y_test)
    perm_test = np.random.permutation(len(y_test))
    x_test = x_test[perm_test]
    y_test = y_test[perm_test]
        
    # preprocessing training samples...
    x_valid = np.array(x_valid ).astype(np.float32)
    x_valid =x_valid .transpose((0,1,2,3))
    x_valid  = preprocess_input(x_valid )    
    y_valid  = np.array(y_valid )
    y_valid  = np.squeeze(y_valid )
    perm_valid  = np.random.permutation(len(y_valid ))
    x_valid  = x_valid[perm_valid]
    y_valid  = y_valid[perm_valid]
        
    # tags are the unique values of the array....     
    tags = np.unique(y_test)    
    #return (x_test, y_test),tags    
    return (x_valid, y_valid),tags
    

def generate_final_mask_from_class(image, logs, patch_h, patch_w,num_rows,num_columns):   
    #final_mask = mask  
    HPF_height,HPF_width, channels = image.shape
    
    mask = np.ones((HPF_height,HPF_width),dtype=int)
    final_mask = np.zeros((HPF_height,HPF_width),dtype=int)
    
    patch_idx = 0    
    num_samples_logs = np.array(logs)     
    for row in range(0,num_rows):
        for column in range(0,num_columns):
            single_log = num_samples_logs[patch_idx,:]
            conf_value = single_log[0]
            index_value = int(single_log[1])                
            if index_value == 0:
                indv_mask = mask [row:row+patch_h, column: column+patch_w]
                indv_mask = (indv_mask/255)*125              
                final_mask [row:row+patch_h, column: column+patch_w ] = indv_mask
            else:
                final_mask [row:row+patch_h, column: column+patch_w] = mask [row:row+patch_h, column: column+patch_w]*0    
    
    return final_mask

def image_from_patches(patches,num_patches, num_rows, num_columns):
    
    patches_size = patches.shape
    
    patch_w = patches_size[1]
    patch_h = patches_size[2]
    
    image_w = patches_size[1]*num_rows
    image_h = patches_size[2]*num_columns
    
    if len(patches_size)>3:
        img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8)
    else:
        img_from_patches = np.zeros((image_w,image_h),dtype=np.uint8)
    
    patch_idx = 0    
    
    for r in range(0,num_rows):
        for c in range(0, num_columns):

            if len(patches_size)>3:
                img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h,:] = patches[patch_idx]
            else:
                img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h] = patches[patch_idx]
            
            patch_idx +=1
    
    img_from_patches = np.array(img_from_patches).astype(np.float32) 
     
    return  img_from_patches  

def image_heatmaps_from_patches(patches,num_patches, num_rows, num_columns):
    
    patches_size = patches.shape
    
    patch_w = patches_size[1]
    patch_h = patches_size[2]
    
    image_w = patches_size[1]*num_rows
    image_h = patches_size[2]*num_columns
    
    if len(patches_size)>3:
        img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8)
    else:
        img_from_patches = np.zeros((image_w,image_h),dtype=np.uint8)
    
    patch_idx = 0    
    
    for r in range(0,num_rows):
        for c in range(0, num_columns):

            if len(patches_size)>3:
                img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h,:] = patches[patch_idx]
            else:
                img_from_patches[r*patch_w: r*patch_w+patch_w,c*patch_h:c*patch_h+patch_h] = patches[patch_idx]
            
            patch_idx +=1
    
    img_from_patches = np.array(img_from_patches).astype(np.float32) 
     
    return  img_from_patches  

