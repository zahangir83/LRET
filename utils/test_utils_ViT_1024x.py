#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:03:13 2024

@author: malom
"""
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pdb
from tensorflow.keras import layers
from tensorflow import keras
import argparse
from os.path import join as join_path
import matplotlib.pyplot as plt
import pathlib
from models import LR_models
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import backend as K
from skimage.transform import resize

from skimage import io
import random
import cv2
import pandas as pd

AUTO = tf.data.AUTOTUNE
#EPOCHS = 150
image_size = (768, 768)
alpha = 0.6
abspath = os.path.dirname(os.path.abspath(__file__))

layer_name_1D = 'FEXL512'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def class_activation_heatmap_of_feature_from_dirs_save_diff_dirs_ViT(scr_dir, model, feature_number, save_dir):
    '''
    Saves the feature heatmap of the given image
    :param image_path:
    :param feature_number:
    :param save_dir:
    :return:
    '''
     
    actual_labels = []
        
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print(dir_name)  
                
            actual_labels.append(dir_name)
                
            if not os.path.isdir("%s/%s"%(save_dir,dir_name)):
                os.makedirs("%s/%s"%(save_dir,dir_name))                
            dst_dir_final = join_path(save_dir,dir_name+'/')    
                
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            total_samples = len(image_list)
            images_files_final = np.array(image_list)
            print('Totla number of samples:', total_samples)                   

            for i, img_name in enumerate(images_files_final): 
                    #print('Copying files for : ',dir_name )
                image_path = join_path(sub_dir_path,img_name)   
                print(image_path)
                img_ext = img_name.split('.')[1]                
                #pdb.set_trace()
                if img_ext =='png':                  
                    check_valid_image_flag = verify_image(image_path)                    
                    if check_valid_image_flag == True:                            
                            # img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            # img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)
                        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
                            #img_array = keras.preprocessing.image.img_to_array(img)
                            #img_array = tf.expand_dims(img_array, 0)  # Create batch axis        
                            
                        img_name_owext = img_name.split('.')[0]
                        img_name_4s = img_name_owext+'_heatmap.png'
                        class_actiation_img_name_4s = img_name_owext+'_heatmap_class_act.png'
                            
                        fm_np_name = img_name_owext+'_fm_vector.npy'
                        cp_np_name = img_name_owext+'_class_prob.npy'

                        # Generate heatmap and extract features and class probs.
                            
                        #fam, fam_class_activation, conv_output_fm, class_probs = _class_activation_heatmap_of_feature_helper_ViT(image_path, feature_number)
                        
                           
                        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
                        img_array = keras.preprocessing.image.img_to_array(img)
                        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                        
                        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
                        final_conv_layer_1D = model.get_layer(name='FEXL512')
                        get_output_fm = K.function([model.layers[0].input], [final_conv_layer_1D.output])
        
                        features_1D = np.squeeze(get_output_fm(img_array)[0]) 
                        conv_output_fm = np.squeeze(features_1D)
        
                        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx      
                        # >>>>>>>>>  check the class probability.....
                        class_probs = model.predict(img_array)  
                        
                            
                        #fam, conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                        # save all representations...
                        final_dst_image = join_path(dst_dir_final, img_name)
                        final_dst_heatmap = join_path(dst_dir_final, img_name_4s)
                        final_dst_heatmap_class_activation = join_path(dst_dir_final, class_actiation_img_name_4s)

                        final_fm_saving_path = join_path(dst_dir_final, fm_np_name)
                        final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)

                            #cv2.imwrite(final_dst_image, img)
                        #cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                        #cv2.imwrite(final_dst_heatmap_class_activation, fam_class_activation.astype(np.uint8, copy=False))
                            
                        np.save(final_fm_saving_path,conv_output_fm)
                        np.save(final_class_prob_saving_path,class_probs)

                            
def read_feature_and_pred_and_merge_them_ViT(scr_dir):
        
    case_ids = []
    y_gt = []
    y_pred = []
    fts_mat = []
    
        #pdb.set_trace()
        
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
    
            #file_list = os.listdir(sub_dir_path) # dir is your directory path
            file_list = [x for x in sorted(os.listdir(sub_dir_path)) if x[-14:] == '_fm_vector.npy']   
            #total_samples = len(file_list)
            files_final = np.array(file_list)
                #print('Totla number of samples:', total_samples) 
    
            for i, file_name in enumerate(files_final): 
                file_path = os.path.join(sub_dir_path,file_name)   
                    #print(file_path)
                file_ext = file_name.split('.')[1]      
                file_name_wo_ext = file_name.split('_fm_vector')[0]#+'.'+file_name.split('.')[1] 
                file_name_wo_ext_fp = file_name.split('_fm_vector')[0]#+'.'
                
                if file_name.endswith('_fm_vector.npy'):
                        
                    case_ids.append(file_name_wo_ext)
                    y_gt.append(dir_name)
                        # read the DL feature representation.....
                    input_fv = np.load(file_path,allow_pickle=True)
                    input_fv = np.squeeze(input_fv)
                    #input_fv_0 = np.mean(input_fv,axis=0)
                    #input_fv_00 = np.mean(input_fv_0,axis=0)
                    exd_input_fv_00 = np.expand_dims(input_fv,axis=0)
                    #fts_mat.append(exd_input_fv_00)
                    # read the predicted probability..........
                    file_path_pred = os.path.join(sub_dir_path,file_name_wo_ext_fp+'_class_prob.npy') 
                    input_pred = np.load(file_path_pred,allow_pickle=True)
                    y_pred.append(input_pred)
        
    case_ids = np.array(case_ids)
    y_gt = np.array(y_gt)
    y_pred_mat = np.squeeze(np.array(y_pred ))
    #fts_mat = np.squeeze(np.array(fts_mat)) 
        
    return case_ids,y_gt,y_pred_mat