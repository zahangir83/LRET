#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:30:37 2020

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
from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage

import scipy.ndimage as ndimage
import pandas as pd 

import nibabel as nib
from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq

alpha = 0.6


def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    
    return img_out 

def dl_feature_saving(feature_saving_path,feature_type,ac_labels,dl_features):
    
    #np.savez('mat.npz', name1=arr1, name2=arr2)
    #data = np.load('mat.npz')
    #print data['name1']
    #print data['name2']

    name = feature_type+'.npy'
    saving_path = join_path(feature_saving_path,name)
    np.save(saving_path,dl_features)
    
    label_name = feature_type+'_ac_labels.npy'
    label_saving_path = join_path(feature_saving_path,label_name)
    np.save(label_saving_path,ac_labels)


def read_feature_and_pred_and_merge_them(scr_dir):
    case_ids = []
    y_gt = []
    y_pred = []
    fts_mat = []

    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           

            file_list = os.listdir(sub_dir_path) # dir is your directory path
            total_samples = len(file_list)
            files_final = np.array(file_list)
            #print('Totla number of samples:', total_samples) 

            for i, file_name in enumerate(files_final): 
                file_path = os.path.join(sub_dir_path,file_name)   
                #print(file_path)
                file_ext = file_name.split('.')[2]      
                file_name_wo_ext = file_name.split('.')[0]+'.'+file_name.split('.')[1] 
                file_name_wo_ext_fp = file_name.split('.')[0]+'.'


                if file_name.endswith('_fm_vector.npy'):
                    
                    case_ids.append(file_name_wo_ext)
                    y_gt.append(dir_name)
                    # read the DL feature representation.....
                    input_fv = np.load(file_path,allow_pickle=True)
                    input_fv_0 = np.mean(input_fv,axis=0)
                    input_fv_00 = np.mean(input_fv_0,axis=0)
                    exd_input_fv_00 = np.expand_dims(input_fv_00,axis=0)
                    fts_mat.append(exd_input_fv_00)
                    
                    # read the predicted probability..........
                    
                    file_path_pred = os.path.join(sub_dir_path,file_name_wo_ext_fp+'png_class_prob.npy') 
                    input_pred = np.load(file_path_pred,allow_pickle=True)
                    y_pred.append(input_pred)
    
    

    case_ids = np.array(case_ids)
    y_gt = np.array(y_gt)
    y_pred_mat = np.squeeze(np.array(y_pred ))
    fts_mat = np.squeeze(np.array(fts_mat)) 
    
    return case_ids,y_gt,y_pred_mat,fts_mat


   

def dl_feature_and_labels_saving(feature_saving_path,feature_type,ac_labels,dl_features):
    
    #np.savez('mat.npz', name1=arr1, name2=arr2)
    #data = np.load('mat.npz')
    #print data['name1']
    #print data['name2']

    name = feature_type+'.npy'
    saving_path = join_path(feature_saving_path,name)
    np.save(saving_path,dl_features)
    
    label_name = feature_type+'_ac_labels.npy'
    label_saving_path = join_path(feature_saving_path,label_name)
    np.save(label_saving_path,ac_labels)


def DL_feature_mat_to_feature_vector(dlf_train_mat, dlf_test_mat):
    
    mat_train = dlf_train_mat.sum(axis=1)
    vec_train = mat_train.sum(axis=1)
    
    mat_test = dlf_test_mat.sum(axis=1)
    vec_test = mat_test.sum(axis=1)
    
    return vec_train,vec_test
    

def generate_heatmaps_and_save_from_DFL(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, dst_dir_final):
    
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
                            
             
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value']
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, 'model_predictions.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)
    

def generate_heatmaps_and_identify_most_important_modality_family_subclasses(case_ids, labels_ac, labels_ac_subclass, y_test, y_hat, original_slices, dl_features, dst_dir_final):
    
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 210
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
        HBPxls = predicted_conf_images_norm*binary_img_bg
        HBPxls_total = HBPxls.sum()

        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        input_slice_flair_4ipt_mean = np.mean(input_slice_flair_4ipt, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())

        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_bg
        HBPxls_flair_total = HBPxls_flair.sum()

        
        flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg
        
        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])
        input_slice_t1_4ipt_mean = np.mean(input_slice_t1_4ipt, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_bg
        HBPxls_t1_total = HBPxls_t1.sum()

        
        t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg
        
        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        input_slice_t1gd_4ipt_mean = np.mean(input_slice_t1gd_4ipt, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_bg
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        input_slice_t2_4ipt_mean = np.mean(input_slice_t2_4ipt, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())

        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_bg
        HBPxls_t2_total = HBPxls_t2.sum()

        
        t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
             
    y_pred_ac = np.array(y_pred_ac)
    
    #pdb.set_trace()
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    #labels_ac_subclass = np.expand_dims(np.array(labels_ac_subclass),axis=-1)
    labels_ac_subclass = np.array(labels_ac_subclass)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','GT-subclasses','PRED_CLASS','CONT_value']
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,labels_ac_subclass,y_pred_ac,y_pred_conf),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, 'model_predictions.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)
    
def generate_heatmaps_and_identify_most_important_modality(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, dst_dir_final):
    
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 190
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
        HBPxls = predicted_conf_images_norm*binary_img_bg
        HBPxls_total = HBPxls.sum()

        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        input_slice_flair_4ipt_mean = np.mean(input_slice_flair_4ipt, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())

        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_bg
        HBPxls_flair_total = HBPxls_flair.sum()

        
        flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])
        input_slice_t1_4ipt_mean = np.mean(input_slice_t1_4ipt, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_bg
        HBPxls_t1_total = HBPxls_t1.sum()

        
        t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        input_slice_t1gd_4ipt_mean = np.mean(input_slice_t1gd_4ipt, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_bg
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg
        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        input_slice_t2_4ipt_mean = np.mean(input_slice_t2_4ipt, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())

        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_bg
        HBPxls_t2_total = HBPxls_t2.sum()

        
        t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
             
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value']
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, 'model_predictions.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    
    
def generate_heatmaps_and_identify_most_important_modality_with_skull_stripping(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, file_name, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    fBR = []
    fTR = []
    t1BR = []
    t1TR = []
    t1gdBR = []
    t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')

        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')

        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')

        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 120
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_flair_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        
        #pdb.set_trace()
        cv2.imwrite(final_dst_heatmap_flair_brain, binary_img_bg_brain_stack*heatmap_flair.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        #flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        #flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        flair_bg_normalized = HBPxls_flair_total/HFPxls_total
        
        fTR.append(flair_fg_normalized)
        fBR.append(flair_bg_normalized)

    
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'mask_fg.png'), binary_img_fg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        #pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        #t1_combined_mask = t1_combined_mask.T
        binary_img_bg_brain = fillhole(t1_combined_mask)

        input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1_total = HBPxls_t1.sum()

        
        #t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        #t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        t1_bg_normalized = HBPxls_t1_total/HFPxls_total
        
        t1TR.append(t1_fg_normalized)
        t1BR.append(t1_bg_normalized)
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        #t1gd_combined_mask = t1gd_combined_mask.T
        #binary_img_bg = t1gd_combined_mask
        input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        #t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        #t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_fg_normalized = HFPxls_t1gd_total/HBPxls_total
        t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        t1gdTR.append(t1gd_fg_normalized)
        t1gdBR.append(t1gd_bg_normalized)
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        #t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        #t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        t2_bg_normalized = HBPxls_t2_total/HFPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
    
    fBR = np.expand_dims(np.array(fBR),axis = -1)
    fTR = np.expand_dims(np.array(fTR),axis = -1)
    t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','FLAIR_BR','FLAIR_TR','T1_BR','T1_TR','T1GD_BR','T1GD_TR','T2_BR','T2_TR']
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,fBR,fTR,t1BR,t1TR,t1gdBR,t1gdTR,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, file_name+'_model_pred.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    #return mat_to_save


def generate_heatmaps_and_identify_most_important_modality_with_skull_stripping_return_conf_mat(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, file_name, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    fBR = []
    fTR = []
    t1BR = []
    t1TR = []
    t1gdBR = []
    t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')
        final_dst_heatmap_flair_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain_norm.png')

        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')
        final_dst_heatmap_t1_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')
        final_dst_heatmap_t1gd_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')
        final_dst_heatmap_t2_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 190
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_flair_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        cv2.imwrite(final_dst_heatmap_flair_brain, binary_img_bg_brain_stack*heatmap_flair.astype(np.uint8, copy=False))
        # Apply MAX-MIN normalization on tissue region...
        #binary_img_bg_brain_stack_norm = (binary_img_bg_brain_stack-binary_img_bg_brain_stack.min())/(binary_img_bg_brain_stack.max()-binary_img_bg_brain_stack.min())
        #cv2.imwrite(final_dst_heatmap_flair_brain_norm, binary_img_bg_brain_stack_norm*heatmap_flair.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        #flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        #flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        flair_bg_normalized = HBPxls_flair_total/HFPxls_total
        
        fTR.append(flair_fg_normalized)
        fBR.append(flair_bg_normalized)

    
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'mask_fg.png'), binary_img_fg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        #pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        flair_combined = flair_combined*binary_img_bg_brain

        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        #t1_combined_mask = t1_combined_mask.T
        binary_img_bg_brain = fillhole(t1_combined_mask)

        input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1_total = HBPxls_t1.sum()

        
        #t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        #t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        t1_bg_normalized = HBPxls_t1_total/HFPxls_total
        
        t1TR.append(t1_fg_normalized)
        t1BR.append(t1_bg_normalized)
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        t1_combined = t1_combined*binary_img_bg_brain
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        #t1gd_combined_mask = t1gd_combined_mask.T
        #binary_img_bg = t1gd_combined_mask
        input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        #t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        #t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_fg_normalized = HFPxls_t1gd_total/HBPxls_total
        t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        t1gdTR.append(t1gd_fg_normalized)
        t1gdBR.append(t1gd_bg_normalized)
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        t1gd_combined = t1gd_combined*binary_img_bg_brain

        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        #t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        #t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        t2_bg_normalized = HBPxls_t2_total/HFPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        t2_combined = t2_combined*binary_img_bg_brain
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
    
    fBR = np.expand_dims(np.array(fBR),axis = -1)
    fTR = np.expand_dims(np.array(fTR),axis = -1)
    t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','FLAIR_BR','FLAIR_TR','T1_BR','T1_TR','T1GD_BR','T1GD_TR','T2_BR','T2_TR',]
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,fBR,fTR,t1BR,t1TR,t1gdBR,t1gdTR,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, file_name+'_model_pred.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    return mat_to_save

def generate_heatmaps_and_identify_most_important_modality_with_skull_stripping_return_conf_mat_final(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, file_name, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    fBR = []
    fTR = []
    t1BR = []
    t1TR = []
    t1gdBR = []
    t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam_mean = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam_mean /= np.max(cam_mean)
        cam = cv2.resize(cam_mean, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image
        final_dst_heatmap_input_level_sp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_hp_input_level.png')
        cv2.imwrite(final_dst_heatmap_input_level_sp, heatmap.astype(np.uint8, copy=False))


        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')
        final_dst_heatmap_flair_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain_norm.png')
        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        
        #  Heatmaps file names
        
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')
        final_dst_heatmap_t1_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')
        final_dst_heatmap_t1gd_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')
        final_dst_heatmap_t2_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 230   #### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> xxxxxxxxxxxxx Thresholod NEED TO BE WORK ON THE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_flair_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        cv2.imwrite(final_dst_heatmap_flair_brain, (binary_img_bg_brain_stack*heatmap_flair).astype(np.uint8, copy=False))
        
        #pdb.set_trace()
        
        # To Generate the heatmaps wrt the tissue region confidences....
        
        #cam_mean = np.mean(conv_output, axis=2)
        
        tissue_region_cam_flair = (binary_img_bg_brain>0)*cam
        tissue_region_heatmaps_norm = (tissue_region_cam_flair-tissue_region_cam_flair.min())/(tissue_region_cam_flair.max()-tissue_region_cam_flair.min())
        
        heatmap_tissue_level = cv2.applyColorMap(np.uint8(255 * tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        final_dst_heatmap_tissue_level_sp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_hp_tissue_level.png')
        cv2.imwrite(final_dst_heatmap_tissue_level_sp, heatmap_tissue_level.astype(np.uint8, copy=False))

        tissue_region_heatmap_norm_flair = cv2.applyColorMap(np.uint8(255 * tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        alpha_tissue = 0.2
        flair_tissue_region_heatmaps_norm = cv2.addWeighted(input_slice_flair, alpha, tissue_region_heatmap_norm_flair, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_flair_brain_norm, flair_tissue_region_heatmaps_norm.astype(np.uint8, copy=False))
        
        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        #flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        #flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        flair_bg_normalized = HBPxls_flair_total/HBPxls_total
        
        fTR.append(flair_fg_normalized)
        fBR.append(flair_bg_normalized)

    
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'mask_fg.png'), binary_img_fg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        flair_combined = flair_combined*binary_img_bg_brain

        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        #t1_combined_mask = t1_combined_mask.T
        binary_img_bg_brain = fillhole(t1_combined_mask)

        input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        ### T1-normalized the tissue region.....
        t1_tissue_region_heatmaps = binary_img_bg_brain*cam
        t1_tissue_region_heatmaps_norm = (t1_tissue_region_heatmaps-t1_tissue_region_heatmaps.min())/(t1_tissue_region_heatmaps.max()-t1_tissue_region_heatmaps.min())
        tissue_region_heatmap_norm_t1 = cv2.applyColorMap(np.uint8(255 * t1_tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        tissue_region_heatmap_norm_t1_norm = cv2.addWeighted(input_slice_t1, alpha, tissue_region_heatmap_norm_t1, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_t1_brain_norm, tissue_region_heatmap_norm_t1_norm.astype(np.uint8, copy=False))
        
        ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1_total = HBPxls_t1.sum()

        
        #t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        #t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        t1_bg_normalized = HBPxls_t1_total/HBPxls_total
        
        t1TR.append(t1_fg_normalized)
        t1BR.append(t1_bg_normalized)
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        t1_combined = t1_combined*binary_img_bg_brain
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        #t1gd_combined_mask = t1gd_combined_mask.T
        #binary_img_bg = t1gd_combined_mask
        input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        ### T1GD  Save the normalized hearmap tissue region....
        #t1gd_tissue_region_heatmaps = binary_img_bg_brain_stack*heatmap_t1gd
        #t1gd_tissue_region_heatmaps_norm = (t1gd_tissue_region_heatmaps-t1gd_tissue_region_heatmaps.min())/(t1gd_tissue_region_heatmaps.max()-t1gd_tissue_region_heatmaps.min())
        #cv2.imwrite(final_dst_heatmap_t1gd_brain_norm, t1gd_tissue_region_heatmaps_norm.astype(np.uint8, copy=False))
        
        t1gd_tissue_region_heatmaps = binary_img_bg_brain*cam
        t1gd_tissue_region_heatmaps_norm = (t1gd_tissue_region_heatmaps-t1gd_tissue_region_heatmaps.min())/(t1gd_tissue_region_heatmaps.max()-t1gd_tissue_region_heatmaps.min())
        tissue_region_heatmap_norm_t1gd = cv2.applyColorMap(np.uint8(255 * t1gd_tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        tissue_region_heatmap_norm_t1gd_norm = cv2.addWeighted(input_slice_t1gd, alpha, tissue_region_heatmap_norm_t1gd, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_t1gd_brain_norm, tissue_region_heatmap_norm_t1gd_norm.astype(np.uint8, copy=False))
        
        
        ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        #t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        #t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_fg_normalized = HFPxls_t1gd_total/HFPxls_total
        t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        t1gdTR.append(t1gd_fg_normalized)
        t1gdBR.append(t1gd_bg_normalized)
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        t1gd_combined = t1gd_combined*binary_img_bg_brain

        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        # T2-normalied heatmap saving tissue regions....        
        t2_tissue_region_heatmaps = binary_img_bg_brain*cam
        t2_tissue_region_heatmaps_norm = (t2_tissue_region_heatmaps-t2_tissue_region_heatmaps.min())/(t2_tissue_region_heatmaps.max()-t2_tissue_region_heatmaps.min())
        tissue_region_heatmap_norm_t2 = cv2.applyColorMap(np.uint8(255 * t2_tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        tissue_region_heatmap_norm_t2_norm = cv2.addWeighted(input_slice_t1, alpha, tissue_region_heatmap_norm_t2, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_t2_brain_norm, tissue_region_heatmap_norm_t2_norm.astype(np.uint8, copy=False))
        
        
        
        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        #t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        #t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        t2_bg_normalized = HBPxls_t2_total/HBPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        t2_combined = t2_combined*binary_img_bg_brain
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
    
    fBR = np.expand_dims(np.array(fBR),axis = -1)
    fTR = np.expand_dims(np.array(fTR),axis = -1)
    t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','FLAIR_BR','FLAIR_TR','T1_BR','T1_TR','T1GD_BR','T1GD_TR','T2_BR','T2_TR',]
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,fBR,fTR,t1BR,t1TR,t1gdBR,t1gdTR,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, file_name+'_model_pred.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    return mat_to_save




def generate_heatmaps_and_identify_most_important_modality_with_skull_stripping_return_conf_mat_using_tumor_masks_final(case_ids, labels_ac, y_test, y_hat, original_slices,  dl_features, masks_test, file_name, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    fBR = []
    fTR = []
    t1BR = []
    t1TR = []
    t1gdBR = []
    t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        input_masks_4modalities = np.squeeze(masks_test[kk])
        
        
        # save example imagees and maskks......
        #pdb.set_trace()
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam_mean = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam_mean /= np.max(cam_mean)
        cam = cv2.resize(cam_mean, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image
        final_dst_heatmap_input_level_sp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_hp_input_level.png')
        cv2.imwrite(final_dst_heatmap_input_level_sp, heatmap.astype(np.uint8, copy=False))


        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')
        final_dst_heatmap_flair_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain_norm.png')
        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        
        #  Heatmaps file names
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')
        final_dst_heatmap_t1_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')
        final_dst_heatmap_t1gd_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')
        final_dst_heatmap_t2_brain_norm = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain_norm.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        
        # Saving the masks for 4 modalites....
        # FLAIR masks ....
        # binary_img_FLAIR = (np.squeeze(input_masks_4modalities[:,:,3])>0)
        # binary_img_fg_FLAIR = (binary_img_FLAIR>0)*1  
        # binary_img_bg_FLAIR = ((np.squeeze(input_masks_4modalities[:,:,3])<=0))*1
    
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_mask.png'), binary_img_fg_FLAIR * 255)
            
        # # T1 masks ....
        # binary_img_T1 = (np.squeeze(input_masks_4modalities[:,:,9])>0)
        # binary_img_fg_T1 = (binary_img_T1>0)*1  
        # binary_img_bg_T1 = ((np.squeeze(input_masks_4modalities[:,:,9])<=0))*1
    
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_mask.png'), binary_img_fg_T1 * 255)
        
        # # T1GD masks ....
        # binary_img_T1GD = (np.squeeze(input_masks_4modalities[:,:,15])>0)
        # binary_img_fg_T1GD = (binary_img_T1GD>0)*1  
        # binary_img_bg_T1GD = ((np.squeeze(input_masks_4modalities[:,:,15])<=0))*1
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_mask.png'), binary_img_fg_T1GD * 255)
        
        # # T2 masks ....
        # binary_img_T2 = (np.squeeze(input_masks_4modalities[:,:,21])>0)
        # binary_img_fg_T2 = (binary_img_T2>0)*1  
        # binary_img_bg_T2 = ((np.squeeze(input_masks_4modalities[:,:,21])<=0))*1
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_mask.png'), binary_img_fg_T2 * 255)
    
    
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DOOONOOOOOT use the segmentation masks generated from classification model confis........<
        threshold_value =230
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> XXxxxxxxxx<<<<<<<<<<<<<<,
        ret,binary_img_thr = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg_thr = (binary_img_thr>125)*1
        invert_binary_img_thr = cv2.bitwise_not(binary_img_thr)     ## Mask for backgournd ....   
        binary_img_bg_thr = (invert_binary_img_thr>125)*1
        
        ## >>>>>>>>>> END <<<<<<<<<<<<<<<<<<<<<<
        
        #pdb.set_trace()
        ## ADD UP all mask for 4 modalities...
        #binary_mask_4modalites = binary_img_fg_FLAIR+binary_img_fg_T1+binary_img_fg_T1GD+binary_img_fg_T2
        
        #binary_img = (np.squeeze(binary_mask_4modalites)>0)
        binary_img = (np.squeeze(input_masks_4modalities[:,:,3])>0)
        #binary_img_btws = np.squeeze(input_masks_4modalities[:,:,3])>0
        binary_img_fg = (binary_img>0)*1
        #invert_binary_img = cv2.bitwise_not((binary_img*255))     ## Mask for backgournd ....   
        #invert_binary_img = invert_binary_img*255
        binary_img_bg = ((np.squeeze(input_masks_4modalities[:,:,3])<=0))*1
        
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_flair_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        cv2.imwrite(final_dst_heatmap_flair_brain, (binary_img_bg_brain_stack*heatmap_flair).astype(np.uint8, copy=False))
        
        #pdb.set_trace()
        
        # To Generate the heatmaps wrt the tissue region confidences....
        
        #cam_mean = np.mean(conv_output, axis=2)
        
        tissue_region_cam_flair = (binary_img_bg_brain>0)*cam
        tissue_region_heatmaps_norm = (tissue_region_cam_flair-tissue_region_cam_flair.min())/(tissue_region_cam_flair.max()-tissue_region_cam_flair.min())
        
        heatmap_tissue_level = cv2.applyColorMap(np.uint8(255 * tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        final_dst_heatmap_tissue_level_sp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_hp_tissue_level.png')
        cv2.imwrite(final_dst_heatmap_tissue_level_sp, heatmap_tissue_level.astype(np.uint8, copy=False))

        tissue_region_heatmap_norm_flair = cv2.applyColorMap(np.uint8(255 * tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        alpha_tissue = 0.2
        flair_tissue_region_heatmaps_norm = cv2.addWeighted(input_slice_flair, alpha, tissue_region_heatmap_norm_flair, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_flair_brain_norm, flair_tissue_region_heatmaps_norm.astype(np.uint8, copy=False))
        
        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        #flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        #flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        flair_bg_normalized = HBPxls_flair_total/HBPxls_total
        
        fTR.append(flair_fg_normalized)
        fBR.append(flair_bg_normalized)

    
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_fg.png'), binary_img_fg * 255)
        
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_thresholded.png'), binary_img_bg_thr * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_fg_thresholded.png'), binary_img_fg_thr * 255)
        
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        #pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        flair_combined = flair_combined*binary_img_bg_brain

        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        #t1_combined_mask = t1_combined_mask.T
        binary_img_bg_brain = fillhole(t1_combined_mask)

        input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        ### T1-normalized the tissue region.....
        t1_tissue_region_heatmaps = binary_img_bg_brain*cam
        t1_tissue_region_heatmaps_norm = (t1_tissue_region_heatmaps-t1_tissue_region_heatmaps.min())/(t1_tissue_region_heatmaps.max()-t1_tissue_region_heatmaps.min())
        tissue_region_heatmap_norm_t1 = cv2.applyColorMap(np.uint8(255 * t1_tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        tissue_region_heatmap_norm_t1_norm = cv2.addWeighted(input_slice_t1, alpha, tissue_region_heatmap_norm_t1, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_t1_brain_norm, tissue_region_heatmap_norm_t1_norm.astype(np.uint8, copy=False))
        
        ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1_total = HBPxls_t1.sum()

        
        #t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        #t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        t1_bg_normalized = HBPxls_t1_total/HBPxls_total
        
        t1TR.append(t1_fg_normalized)
        t1BR.append(t1_bg_normalized)
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        t1_combined = t1_combined*binary_img_bg_brain
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        #t1gd_combined_mask = t1gd_combined_mask.T
        #binary_img_bg = t1gd_combined_mask
        input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        ### T1GD  Save the normalized hearmap tissue region....
        #t1gd_tissue_region_heatmaps = binary_img_bg_brain_stack*heatmap_t1gd
        #t1gd_tissue_region_heatmaps_norm = (t1gd_tissue_region_heatmaps-t1gd_tissue_region_heatmaps.min())/(t1gd_tissue_region_heatmaps.max()-t1gd_tissue_region_heatmaps.min())
        #cv2.imwrite(final_dst_heatmap_t1gd_brain_norm, t1gd_tissue_region_heatmaps_norm.astype(np.uint8, copy=False))
        
        t1gd_tissue_region_heatmaps = binary_img_bg_brain*cam
        t1gd_tissue_region_heatmaps_norm = (t1gd_tissue_region_heatmaps-t1gd_tissue_region_heatmaps.min())/(t1gd_tissue_region_heatmaps.max()-t1gd_tissue_region_heatmaps.min())
        tissue_region_heatmap_norm_t1gd = cv2.applyColorMap(np.uint8(255 * t1gd_tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        tissue_region_heatmap_norm_t1gd_norm = cv2.addWeighted(input_slice_t1gd, alpha, tissue_region_heatmap_norm_t1gd, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_t1gd_brain_norm, tissue_region_heatmap_norm_t1gd_norm.astype(np.uint8, copy=False))
        
        
        ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        #t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        #t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_fg_normalized = HFPxls_t1gd_total/HFPxls_total
        t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        t1gdTR.append(t1gd_fg_normalized)
        t1gdBR.append(t1gd_bg_normalized)
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        t1gd_combined = t1gd_combined*binary_img_bg_brain

        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        # T2-normalied heatmap saving tissue regions....        
        t2_tissue_region_heatmaps = binary_img_bg_brain*cam
        t2_tissue_region_heatmaps_norm = (t2_tissue_region_heatmaps-t2_tissue_region_heatmaps.min())/(t2_tissue_region_heatmaps.max()-t2_tissue_region_heatmaps.min())
        tissue_region_heatmap_norm_t2 = cv2.applyColorMap(np.uint8(255 * t2_tissue_region_heatmaps_norm), cv2.COLORMAP_JET)
        tissue_region_heatmap_norm_t2_norm = cv2.addWeighted(input_slice_t1, alpha, tissue_region_heatmap_norm_t2, 1 - alpha_tissue, 0)
        cv2.imwrite(final_dst_heatmap_t2_brain_norm, tissue_region_heatmap_norm_t2_norm.astype(np.uint8, copy=False))
        
        
        
        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        #t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        #t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        t2_bg_normalized = HBPxls_t2_total/HBPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        t2_combined = t2_combined*binary_img_bg_brain
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. EXTENDED CODES  FOR REBUTTAL  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        #pdb.set_trace()
        
        FLAIR_norm_img = cv2.normalize(src=input_slice_4modalities[:,:,2:5], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        T2_norm_img = cv2.normalize(src=input_slice_4modalities[:,:,23:26], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        # create cyan image
        cyan = np.full_like(FLAIR_norm_img,(255,255,0))

        # add cyan to img and save as new image
        blend = 0.75
        FLAIR_img_cyan = cv2.addWeighted(FLAIR_norm_img, blend, cyan, 1-blend, 0)
        T2_img_cyan = cv2.addWeighted(T2_norm_img, blend, cyan, 1-blend, 0)

        mask_90CF_idv = (binary_img_fg_thr>0) * 255
        
        mask_90CF = []
        for kk in range(3):
            mask_90CF.append(mask_90CF_idv)
        
        mask_90CF = np.array(mask_90CF)
        mask_90CF = np.moveaxis(mask_90CF, 0, -1)
        
        # combine img and img_cyan using mask
        FLAIR_result = np.where(mask_90CF==255, FLAIR_img_cyan, FLAIR_norm_img)
        T2_result = np.where(mask_90CF==255, T2_img_cyan, T2_norm_img)


        final_dst_heatmap_flair_brain_90CF = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_90CF.png')
        cv2.imwrite(final_dst_heatmap_flair_brain_90CF, FLAIR_result.astype(np.uint8, copy=False))


        final_dst_heatmap_t2_brain_90CF = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_90CF.png')
        cv2.imwrite(final_dst_heatmap_t2_brain_90CF, T2_result.astype(np.uint8, copy=False))


        
    
    fBR = np.expand_dims(np.array(fBR),axis = -1)
    fTR = np.expand_dims(np.array(fTR),axis = -1)
    t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','FLAIR_BR','FLAIR_TR','T1_BR','T1_TR','T1GD_BR','T1GD_TR','T2_BR','T2_TR',]
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,fBR,fTR,t1BR,t1TR,t1gdBR,t1gdTR,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, file_name+'_model_pred.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    return mat_to_save




def generate_heatmaps_and_identify_most_important_modality_with_skull_stripping_return_conf_mat_within_tissue_regions(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, file_name, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    fBR = []
    fTR = []
    t1BR = []
    t1TR = []
    t1gdBR = []
    t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')

        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')

        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')

        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 190
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_flair_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        
        #pdb.set_trace()
        cv2.imwrite(final_dst_heatmap_flair_brain, binary_img_bg_brain_stack*heatmap_flair.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        #flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        #flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        flair_bg_normalized = HBPxls_flair_total/HBPxls_total
        
        fTR.append(flair_fg_normalized)
        fBR.append(flair_bg_normalized)

    
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'mask_fg.png'), binary_img_fg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        #pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        flair_combined = flair_combined*binary_img_bg_brain

        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        #t1_combined_mask = t1_combined_mask.T
        binary_img_bg_brain = fillhole(t1_combined_mask)

        input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1_total = HBPxls_t1.sum()

        
        #t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        #t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        t1_bg_normalized = HBPxls_t1_total/HBPxls_total
        
        t1TR.append(t1_fg_normalized)
        t1BR.append(t1_bg_normalized)
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        t1_combined = t1_combined*binary_img_bg_brain
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        #t1gd_combined_mask = t1gd_combined_mask.T
        #binary_img_bg = t1gd_combined_mask
        input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        #t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        #t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        t1gd_fg_normalized = HFPxls_t1gd_total/HFPxls_total
        t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        t1gdTR.append(t1gd_fg_normalized)
        t1gdBR.append(t1gd_bg_normalized)
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        t1gd_combined = t1gd_combined*binary_img_bg_brain

        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        #t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        #t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        t2_bg_normalized = HBPxls_t2_total/HBPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        t2_combined = t2_combined*binary_img_bg_brain
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
    
    fBR = np.expand_dims(np.array(fBR),axis = -1)
    fTR = np.expand_dims(np.array(fTR),axis = -1)
    t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','FLAIR_BR','FLAIR_TR','T1_BR','T1_TR','T1GD_BR','T1GD_TR','T2_BR','T2_TR',]
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,fBR,fTR,t1BR,t1TR,t1gdBR,t1gdTR,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, file_name+'_model_pred.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    return mat_to_save

def generate_heatmaps_and_identify_most_important_modality_with_skull_stripping_1mtdp(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    fBR = []
    fTR = []
    t1BR = []
    t1TR = []
    t1gdBR = []
    t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        input_slice_flair = 255 * input_slice_flair # Now scale by 255
        input_slice_flair = input_slice_flair.astype(np.uint8)
        
        input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,20:23])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')

        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')

        cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')

        cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,21], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 120
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_flair_4ipt = np.squeeze(input_slice_4modalities[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_flair_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        
        #pdb.set_trace()
        cv2.imwrite(final_dst_heatmap_flair_brain, binary_img_bg_brain_stack*heatmap_flair.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        flair_fg_normalized = 1-HFPxls_total/HFPxls_flair_total
        flair_bg_normalized = 1-HBPxls_total/HBPxls_flair_total
        
        #flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        #flair_bg_normalized = HBPxls_flair_total/HFPxls_total
        
        fTR.append(flair_fg_normalized)
        fBR.append(flair_bg_normalized)

    
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'mask_fg.png'), binary_img_fg * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        #pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        #t1_combined_mask = t1_combined_mask.T
        binary_img_bg_brain = fillhole(t1_combined_mask)

        input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        HFPxls_t1_total = HFPxls_t1.sum()
        
        HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1_total = HBPxls_t1.sum()

        
        t1_fg_normalized = 1-HFPxls_total/HFPxls_t1_total
        t1_bg_normalized = 1-HBPxls_total/HBPxls_t1_total
        
        #t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        #t1_bg_normalized = HBPxls_t1_total/HFPxls_total
        
        t1TR.append(t1_fg_normalized)
        t1BR.append(t1_bg_normalized)
        
        t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        #t1gd_combined_mask = t1gd_combined_mask.T
        #binary_img_bg = t1gd_combined_mask
        input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        t1gd_fg_normalized = 1-HFPxls_total/HFPxls_t1gd_total
        t1gd_bg_normalized = 1-HBPxls_total/HBPxls_t1gd_total
        
        #t1gd_fg_normalized = HFPxls_t1gd_total/HBPxls_total
        #t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        t1gdTR.append(t1gd_fg_normalized)
        t1gdBR.append(t1gd_bg_normalized)
        
        t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_4modalities[:,:,21:28])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        t2_fg_normalized = 1-HFPxls_total/HFPxls_t2_total
        t2_bg_normalized = 1-HBPxls_total/HBPxls_t2_total
        
        #t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        #t2_bg_normalized = HBPxls_t2_total/HFPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
    
    fBR = np.expand_dims(np.array(fBR),axis = -1)
    fTR = np.expand_dims(np.array(fTR),axis = -1)
    t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','FLAIR_BR','FLAIR_TR','T1_BR','T1_TR','T1GD_BR','T1GD_TR','T2_BR','T2_TR',]
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,fBR,fTR,t1BR,t1TR,t1gdBR,t1gdTR,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, 'model_predictions.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    

def generate_heatmaps_for_single_modality_from_DFL(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, dst_dir_final):
    
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_4modalities = np.squeeze(original_slices[kk])
        
        # input_slice_flair = np.squeeze(input_slice_4modalities[:,:,2:5])
        # input_slice_flair = input_slice_flair.astype(np.float64) / input_slice_flair.max() # normalize the data to 0 - 1
        # input_slice_flair = 255 * input_slice_flair # Now scale by 255
        # input_slice_flair = input_slice_flair.astype(np.uint8)
        
        # input_slice_t1 = np.squeeze(input_slice_4modalities[:,:,8:11])
        # input_slice_t1 = input_slice_t1.astype(np.float64) / input_slice_t1.max() # normalize the data to 0 - 1
        # input_slice_t1 = 255 * input_slice_t1 # Now scale by 255
        # input_slice_t1 = input_slice_t1.astype(np.uint8)
        
        # input_slice_t1gd = np.squeeze(input_slice_4modalities[:,:,14:17])
        # input_slice_t1gd = input_slice_t1gd.astype(np.float64) / input_slice_t1gd.max() # normalize the data to 0 - 1
        # input_slice_t1gd = 255 * input_slice_t1gd # Now scale by 255
        # input_slice_t1gd = input_slice_t1gd.astype(np.uint8)
        
        input_slice_t2 = np.squeeze(input_slice_4modalities[:,:,2:5])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
       # heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
       # heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        #heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
       #  #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
       #  final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
       #  final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
       #  #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
       #  #pdb.set_trace()
        
       #  cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

       # # cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
       #  final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
       #  final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
       #  cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
       #  #cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
       #  final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
       #  final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
       #  cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
       #  #cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
                            
             
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value']
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, 'model_predictions.csv')

    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)
    
    
def generate_heatmaps_and_identify_most_important_T2_modality_with_skull_stripping_return_conf_mat(case_ids, labels_ac, y_test, y_hat, original_slices, dl_features, file_name, dst_dir_final):
    
    ## Load the skull stripping model.... first....
    # model_loading_path = model_path+'model.h5'
    # model = keras.models.load_model(model_loading_path)
    # model.summary() 
    
    ## Take input shape and others....
    _,img_h,img_w,_ = original_slices.shape
    num_cases,H_in,W_in,NFmaps = dl_features.shape
    
       
    if len(y_test.shape)>1:
        y_test = np.argmax(y_test, axis=1)
       
    y_pred_conf = np.max(y_hat,axis=1)         
    if len(y_hat.shape)>1:
        y_pred = np.argmax(y_hat, axis=1)
    
    unique_encode = np.unique(y_test)
    unique_ac_label = np.unique(labels_ac)
    
    #idx_zero = np.where(y_test == 0)
    #label_zero_ac = y_ac_test[idx_zero]
    #skip_by = 6
    y_pred_ac = []

    # fBR = []
    # fTR = []
    # t1BR = []
    # t1TR = []
    # t1gdBR = []
    # t1gdTR = []
    t2BR = []
    t2TR = []
    
    for kk in range(num_cases):
        
        idv_label = y_pred[kk]
        
        if (idv_label == unique_encode[0]):
            y_pred_ac.append(unique_ac_label[0])
        else: 
            y_pred_ac.append(unique_ac_label[1])
    
        idv_case_id = case_ids[kk]
        idv_case_label = labels_ac[kk]
        input_slice_T2modality = np.squeeze(original_slices[kk])
        

        
        input_slice_t2 = np.squeeze(input_slice_T2modality[:,:,2:5])
        input_slice_t2 = input_slice_t2.astype(np.float64) / input_slice_t2.max() # normalize the data to 0 - 1
        input_slice_t2 = 255 * input_slice_t2 # Now scale by 255
        input_slice_t2 = input_slice_t2.astype(np.uint8)

        conv_output = np.squeeze(dl_features[kk])
        
        #H_idv, W_idv, NFmaps_idv = conv_output.shape 
        #average_fm = np.mean(array, axis=0)).
        #rand_samples = random.sample(range(0,NFmaps_idv),how_many_samples)
            # generate heatmap
        #cam = conv_output[:, :, feature_number - 1] 
        
        cam = np.mean(conv_output, axis=2)
        #cam = conv_output
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img_h,img_w))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image

        #pdb.set_trace()
        
        ## Generate the heatmaps and saving the images......
        
        # heatmap_flair = cv2.addWeighted(input_slice_flair, alpha, heatmap, 1 - alpha, 0)
        # heatmap_t1 = cv2.addWeighted(input_slice_t1, alpha, heatmap, 1 - alpha, 0)
        # heatmap_t1gd = cv2.addWeighted(input_slice_t1gd, alpha, heatmap, 1 - alpha, 0)
        heatmap_t2 = cv2.addWeighted(input_slice_t2, alpha, heatmap, 1 - alpha, 0)
                      
        #fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
        # final_dst_image_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair.png')
        # final_dst_heatmap_flair = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp.png')
        # final_dst_heatmap_flair_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_hp_brain.png')

        #cv2.imwrite(final_dst_image_flair, input_slice_flair[:,:,1])
        #pdb.set_trace()
        
        # cv2.imwrite(final_dst_image_flair, cv2.normalize(src=input_slice_4modalities[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        # cv2.imwrite(final_dst_heatmap_flair, heatmap_flair.astype(np.uint8, copy=False))
        
        # final_dst_image_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1.png')
        # final_dst_heatmap_t1 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp.png')
        # final_dst_heatmap_t1_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_hp_brain.png')

        # cv2.imwrite(final_dst_image_t1, cv2.normalize(src=input_slice_4modalities[:,:,9], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        # cv2.imwrite(final_dst_heatmap_t1, heatmap_t1.astype(np.uint8, copy=False))
        
        # final_dst_image_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd.png')
        # final_dst_heatmap_t1gd = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp.png')
        # final_dst_heatmap_t1gd_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_hp_brain.png')

        # cv2.imwrite(final_dst_image_t1gd, cv2.normalize(src=input_slice_4modalities[:,:,15], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        # cv2.imwrite(final_dst_heatmap_t1gd, heatmap_t1gd.astype(np.uint8, copy=False))
        
        final_dst_image_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2.png')
        final_dst_heatmap_t2 = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp.png')
        final_dst_heatmap_t2_brain = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_hp_brain.png')

        cv2.imwrite(final_dst_image_t2, cv2.normalize(src=input_slice_T2modality[:,:,3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        cv2.imwrite(final_dst_heatmap_t2, heatmap_t2.astype(np.uint8, copy=False))
        
        ## Find the best modality from the heatmaps......   

        # Create the binary image from the HEATMAPS images....
        predicted_conf_images = np.uint8(255 * cam)
        predicted_conf_images_norm = predicted_conf_images/predicted_conf_images.max()
        
        threshold_value = 190
        ret,binary_img = cv2.threshold(predicted_conf_images,threshold_value,255,cv2.THRESH_BINARY)    # Mask for the forground region....(expected the tumor regions)
        binary_img_fg = (binary_img>125)*1
        invert_binary_img = cv2.bitwise_not(binary_img)     ## Mask for backgournd ....   
        binary_img_bg = (invert_binary_img>125)*1
        
        # masking the heatmaps coefficent with binray mask
        HFPxls = predicted_conf_images_norm*binary_img_fg
        HFPxls_total = HFPxls.sum()        
     
        #pdb.set_trace()        
        # Generating the importantancy for flair modality ...... 
        input_slice_t2_all_4ipt = np.squeeze(input_slice_T2modality[:,:,0:7])
        
        ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        b0_mask_flair, mask_flair = median_otsu(input_slice_t2_all_4ipt, median_radius=2, numpass=1)
        flair_combined_mask =((mask_flair.sum(axis=2))>0)*1
        #flair_combined_mask = flair_combined_mask.T
        binary_img_bg_brain = fillhole(flair_combined_mask)
        binary_img_bg_brain_stack = np.stack([binary_img_bg_brain,binary_img_bg_brain,binary_img_bg_brain],axis=2)
        binary_img_bg_brain_stack = (binary_img_bg_brain_stack>0)
        
        input_slice_flair_4ipt_mean = np.mean(b0_mask_flair, axis=2)
        input_slice_flair_4ipt_mean = (input_slice_flair_4ipt_mean-input_slice_flair_4ipt_mean.min())/(input_slice_flair_4ipt_mean.max()-input_slice_flair_4ipt_mean.min())
        
        ## Save heatmaps only for the brain region.... 
        
        #pdb.set_trace()
        #cv2.imwrite(final_dst_heatmap_flair_brain, binary_img_bg_brain_stack*heatmap_flair.astype(np.uint8, copy=False))

        ## >>>>>>>>>>>>>>>>>>>>>.. END OF BRAIN MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Generate binary mask except tumor ...
        binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)

        # calculate the values for background pixels .......
        HBPxls = predicted_conf_images_norm*binary_img_brain_except_tumor
        HBPxls_total = HBPxls.sum() 
        
        HFPxls_flair = input_slice_flair_4ipt_mean*binary_img_fg
        HFPxls_flair_total = HFPxls_flair.sum()
        HBPxls_flair = input_slice_flair_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_flair_total = HBPxls_flair.sum()

        #flair_fg_normalized = HFPxls_total/HFPxls_flair_total
        #flair_bg_normalized = HBPxls_total/HBPxls_flair_total
        
        # flair_fg_normalized = HFPxls_flair_total/HFPxls_total
        # flair_bg_normalized = HBPxls_flair_total/HFPxls_total
        
        # fTR.append(flair_fg_normalized)
        # fBR.append(flair_bg_normalized)

    
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg.png'), binary_img_bg * 255)
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'mask_fg.png'), binary_img_fg * 255)
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain_except_tumor.png'), binary_img_brain_except_tumor * 255)
        # cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_bg_brain.png'), binary_img_bg_brain * 255)
        
        #pdb.set_trace()

        non_tissue_mask = cv2.bitwise_not(binary_img_bg_brain)+256
        cv2.imwrite(join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_mask_non_tissue.png'), non_tissue_mask * 255)

        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_brain_except_tumor
        #flair_combined = cv2.bitwise_and(flair_combined, binary_img_bg_brain)
        #flair_combined = flair_fg_normalized*binary_img_fg+flair_bg_normalized*binary_img_bg*0

        #flair_expl_final_image = cv2.applyColorMap(np.uint8(255 * flair_combined), cv2.COLORMAP_JET)
        #final_dst_image_flair_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_flair_pxlimp.png')
        #cv2.imwrite(final_dst_image_flair_exp, flair_expl_final_image.astype(np.uint8, copy=False))

        # # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> T1 modality ...... 
        # input_slice_t1_4ipt = np.squeeze(input_slice_4modalities[:,:,7:14])

        # ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        # b0_mask_t1, mask_t1 = median_otsu(input_slice_t1_4ipt, median_radius=2, numpass=1)
        # t1_combined_mask =((mask_t1.sum(axis=2))>0)*1
        # #t1_combined_mask = t1_combined_mask.T
        # binary_img_bg_brain = fillhole(t1_combined_mask)

        # input_slice_t1_4ipt_mean = np.mean(b0_mask_t1, axis=2)
        # input_slice_t1_4ipt_mean = (input_slice_t1_4ipt_mean-input_slice_t1_4ipt_mean.min())/(input_slice_t1_4ipt_mean.max()-input_slice_t1_4ipt_mean.min())

        # cv2.imwrite(final_dst_heatmap_t1_brain, binary_img_bg_brain_stack*heatmap_t1.astype(np.uint8, copy=False))

        # ## >>>>>>>>>>>>>>>>>>>>>.. END OF MASK EXTRACTION <<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # # Generate binary mask except tumor ...
        # #binary_img_brain_except_tumor= cv2.bitwise_and(binary_img_bg,binary_img_bg_brain)


        # HFPxls_t1 = input_slice_t1_4ipt_mean*binary_img_fg
        # HFPxls_t1_total = HFPxls_t1.sum()
        
        # HBPxls_t1 = input_slice_t1_4ipt_mean*binary_img_brain_except_tumor
        # HBPxls_t1_total = HBPxls_t1.sum()

        
        # #t1_fg_normalized = HFPxls_total/HFPxls_t1_total
        # #t1_bg_normalized = HBPxls_total/HBPxls_t1_total
        
        # t1_fg_normalized = HFPxls_t1_total/HFPxls_total
        # t1_bg_normalized = HBPxls_t1_total/HFPxls_total
        
        # t1TR.append(t1_fg_normalized)
        # t1BR.append(t1_bg_normalized)
        
        # t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_brain_except_tumor
        # #t1_combined = t1_fg_normalized*binary_img_fg+t1_bg_normalized*binary_img_bg*0

        # t1_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1_combined), cv2.COLORMAP_JET)
        # final_dst_image_t1_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1_pxlimp.png')
        # cv2.imwrite(final_dst_image_t1_exp, t1_expl_final_image.astype(np.uint8, copy=False))
        
        # # Generating the importantancy for >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>T1gd modality ...... 
        # input_slice_t1gd_4ipt = np.squeeze(input_slice_4modalities[:,:,14:21])
        # ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # b0_mask_t1gd, mask_t1gd = median_otsu(input_slice_t1gd_4ipt, median_radius=2, numpass=1)
        # t1gd_combined_mask =((mask_t1gd.sum(axis=2))>0)*1
        # #t1gd_combined_mask = t1gd_combined_mask.T
        # #binary_img_bg = t1gd_combined_mask
        # input_slice_t1gd_4ipt_mean = np.mean(b0_mask_t1gd, axis=2)
        # input_slice_t1gd_4ipt_mean = (input_slice_t1gd_4ipt_mean-input_slice_t1gd_4ipt_mean.min())/(input_slice_t1gd_4ipt_mean.max()-input_slice_t1gd_4ipt_mean.min())

        # cv2.imwrite(final_dst_heatmap_t1gd_brain, binary_img_bg_brain_stack*heatmap_t1gd.astype(np.uint8, copy=False))

        # ### >>>>>>>>>>>>>>>>>>> Process END <<<<<<<<<<<<<<<<<<<<<
        # HFPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_fg
        # HFPxls_t1gd_total = HFPxls_t1gd.sum()
        
        # HBPxls_t1gd = input_slice_t1gd_4ipt_mean*binary_img_brain_except_tumor
        # HBPxls_t1gd_total = HBPxls_t1gd.sum()

        
        # #t1gd_fg_normalized = HFPxls_total/HFPxls_t1gd_total
        # #t1gd_bg_normalized = HBPxls_total/HBPxls_t1gd_total
        
        # t1gd_fg_normalized = HFPxls_t1gd_total/HBPxls_total
        # t1gd_bg_normalized = HBPxls_t1gd_total/HBPxls_total
        
        # t1gdTR.append(t1gd_fg_normalized)
        # t1gdBR.append(t1gd_bg_normalized)
        
        # t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_brain_except_tumor
        # #t1gd_combined = t1gd_fg_normalized*binary_img_fg+t1gd_bg_normalized*binary_img_bg*0
        
        # t1gd_expl_final_image = cv2.applyColorMap(np.uint8(255 * t1gd_combined), cv2.COLORMAP_JET)
        # final_dst_image_t1gd_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t1gd_pxlimp.png')
        # cv2.imwrite(final_dst_image_t1gd_exp, t1gd_expl_final_image.astype(np.uint8, copy=False))
        
        
        # Generating the importantancy for T2 modality ...... 
        input_slice_t2_4ipt = np.squeeze(input_slice_T2modality[:,:,2:5])
        
         ####>>>>>>>>>>>>>>>>>>>>> APPLY THE SKULL SEGMENTATION MODEL HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        b0_mask_t2, mask_t2 = median_otsu(input_slice_t2_4ipt, median_radius=2, numpass=1)
        t2_combined_mask =((mask_t2.sum(axis=2))>0)*1
        #t2_combined_mask = t2_combined_mask.T
        #binary_img_bg = t2_combined_mask
        
        input_slice_t2_4ipt_mean = np.mean(b0_mask_t2, axis=2)
        input_slice_t2_4ipt_mean = (input_slice_t2_4ipt_mean-input_slice_t2_4ipt_mean.min())/(input_slice_t2_4ipt_mean.max()-input_slice_t2_4ipt_mean.min())
        
        cv2.imwrite(final_dst_heatmap_t2_brain, binary_img_bg_brain_stack*heatmap_t2.astype(np.uint8, copy=False))

        ### End process..............
        
        HFPxls_t2 = input_slice_t2_4ipt_mean*binary_img_fg
        HFPxls_t2_total = HFPxls_t2.sum()
        
        HBPxls_t2 = input_slice_t2_4ipt_mean*binary_img_brain_except_tumor
        HBPxls_t2_total = HBPxls_t2.sum()

        
        #t2_fg_normalized = HFPxls_total/HFPxls_t2_total
        #t2_bg_normalized = HBPxls_total/HBPxls_t2_total
        
        t2_fg_normalized = HFPxls_t2_total/HFPxls_total
        t2_bg_normalized = HBPxls_t2_total/HFPxls_total
        
        t2TR.append(t2_fg_normalized)
        t2BR.append(t2_bg_normalized)
        
        t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_brain_except_tumor
        t2_combined = t2_combined*binary_img_bg_brain

        #t2_combined = t2_fg_normalized*binary_img_fg+t2_bg_normalized*binary_img_bg*0
        
        ft2_expl_final_image = cv2.applyColorMap(np.uint8(255 * t2_combined), cv2.COLORMAP_JET)
        final_dst_image_t2_exp = join_path(dst_dir_final, idv_case_id+'_'+idv_case_label+'_t2_pxlimp.png')
        cv2.imwrite(final_dst_image_t2_exp, ft2_expl_final_image.astype(np.uint8, copy=False))
        
        
    
    #fBR = np.expand_dims(np.array(fBR),axis = -1)
    #fTR = np.expand_dims(np.array(fTR),axis = -1)
    #t1BR = np.expand_dims(np.array(t1BR),axis = -1)
    #t1TR = np.expand_dims(np.array(t1TR),axis = -1)
    #t1gdBR = np.expand_dims(np.array(t1gdBR),axis = -1)
    #t1gdTR = np.expand_dims(np.array(t1gdTR),axis = -1)
    t2BR = np.expand_dims(np.array(t2BR),axis = -1)
    t2TR = np.expand_dims(np.array(t2TR),axis = -1)

         
    y_pred_ac = np.array(y_pred_ac)
    
    ## Expand the dimension to save the file....
    case_ids = np.expand_dims(case_ids,axis=-1)
    labels_ac = np.expand_dims(labels_ac,axis=-1)
    y_pred_ac = np.expand_dims(y_pred_ac,axis=-1)
    y_pred_conf = np.expand_dims(y_pred_conf, axis=-1)
    
    name_row = ['Case_IDs','GT','PRED_CLASS','CONT_value','T2_BR','T2_TR',]
    name_row_exp = np.expand_dims(name_row, axis = 0)
    mat_to_save = np.concatenate((case_ids,labels_ac,y_pred_ac,y_pred_conf,t2BR,t2TR),axis = 1)
    mat_to_save = np.array(mat_to_save)
    
    #pdb.set_trace()  
    mat_to_save_final = np.concatenate((name_row_exp,mat_to_save),axis = 0)

    #csv_saving_path = join_path(dst_dir_final, unique_ac_label[0]+'_vs_'+unique_ac_label[1]+'_outputs.csv')
    csv_saving_path = join_path(dst_dir_final, file_name+'_model_pred.csv')
    pd.DataFrame(mat_to_save_final).to_csv(csv_saving_path)    
    
    return mat_to_save    