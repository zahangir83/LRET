#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:34:50 2024

@author: malom
"""

import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from os.path import join as join_path
import keras.backend as K
from keras.models import load_model
#from utils import utils as utils
#from utils import helpers as helpers
#from utils import heatmap_utils as htm_utils
#from utils import svs_utils as svs_utils
#from builders import model_builder
#import hpf_patches_utils_ts
#from wsi_utils import svs_utils_final_one as svs_utils
#import svs_utils_final_one as svs_utils
#from wsi_utils import svs_utils_final_one_v2 as svs_utils_v2

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
    
from sklearn.decomposition import PCA

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt

#from models import R2UNet as seg_models
from skimage import io

#import seaborn as sn
import shutil
import json
from PIL import Image
#import scipy.ndimage as ndimage
kernel = np.ones((3,3), np.uint8) 
font = cv2.FONT_HERSHEY_SIMPLEX
from tensorflow import keras
import random
import shutil

import openslide
#from scipy.misc import imsave, imresize
from openslide import open_slide # http://openslide.org/api/python/

import pdb

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
    
valid_images = ['.svs','.jpg','.png']

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def most_frequent(List): 
    return max(set(List), key = List.count) 
  
def random_number_generator_in_range(number_samples, how_many_samples, top_range):
    random_vector = []
    for k in range(number_samples):
         rand_samples = random.sample(range(0,top_range),how_many_samples)
         random_vector.append(rand_samples)
    return random_vector



def read_and_preprocess_img(path, size=(224,224)):
    # print the path.....
    #print('Image path : ', path)
    
    #pdb.set_trace()
    #if path.split('.')[-1] == 'jpg' or 'png':
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def read_and_wo_preprocess_img(path, size=(224,224)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x
    
def read_numpy_files_for_patches_of_WSI(numpy_file_path):
    

    image_npy = [x for x in sorted(os.listdir(numpy_file_path)) if x[-7:] == '_fv.npy']
    
    WSI_ID = []
    patch_IDS = []
    fv_all = []
    class_name_all = []
    cfv_all = []
    
    pdb.set_trace()    
    
    for i, f in enumerate(image_npy):
        #ext = os.path.splitext(f)[1]
        #if ext.lower() not in valid_images:
        #    continue
        
        file_name,file_ext = os.path.splitext(f)  

        actual_file_name = file_name.split('_fv')[0]   
        patch_IDS.append(actual_file_name)
        print('Workig for ',actual_file_name)
        
        if i ==0:
            WSI_ID = actual_file_name.split('_')[0]
               
        fv_file_name = file_name+'.npy'
        class_file_name = actual_file_name+'_cls_name.npy'
        confv_file_name = actual_file_name+'_conf_value.npy'
        
        fv_sub_dir = join_path(numpy_file_path,fv_file_name)
        class_sub_dir = join_path(numpy_file_path,class_file_name)
        cfv_sub_dir = join_path(numpy_file_path,confv_file_name)
        
        fv_file = np.load(fv_sub_dir)
        class_file = np.load(class_sub_dir)
        cfv_file = np.load(cfv_sub_dir)
        class_name_all.append(class_file)
        
        fv_file = np.expand_dims(fv_file,axis=0)
        #fv_file = np.expand_dims(fv_file,axis=0)
       
        if i==0:
            fv_all = fv_file
            #class_name_all = class_file
            cfv_all = cfv_file
        else:
            fv_all = np.concatenate((fv_all, fv_file),axis =0)
            #class_name_all = np.concatenate((class_name_all, class_file),axis =1)
            cfv_all = np.concatenate((cfv_all,cfv_file),axis =0)
    
    #pdb.set_trace()
    
    patch_IDS = np.array(patch_IDS)
    fv_all = np.array(fv_all)
    class_name_all = np.array(class_name_all)
    cfv_all = np.array(cfv_all)
        
    np.save(join_path(numpy_file_path,WSI_ID+'_IDs_all.npy'),patch_IDS)        
    np.save(join_path(numpy_file_path,WSI_ID+'_fvs_all.npy'),fv_all) 
    np.save(join_path(numpy_file_path,WSI_ID+'_names_all.npy'),class_name_all)  
    np.save(join_path(numpy_file_path,WSI_ID+'_confvs_all.npy'),cfv_all)  


def CAM_heatmap_with_PCA(CAM_input, resized_img, idv_wsi_name, testing_results_wsi_output_):
        
    # Apply PCA separately to each channel
         
    reconstructed_image = []
        
    for i in range(3):  # Loop over the 3 color channels
            image_2D = CAM_input[:, :, i]
            pca = PCA(n_components=min(image_2D.shape))  # Ensure valid n_components
            transformed = pca.fit_transform(image_2D)  # Perform PCA
            reconstructed_ = pca.inverse_transform(transformed)  # Reconstruct channel
            reconstructed_image.append(reconstructed_)
        
    # Stack the reconstructed channels
    reconstructed_image = np.stack(reconstructed_image, axis=-1)
    reconstructed_heatmap_pca = 255*(np.clip(reconstructed_image, 0, 1))
        
    # Ensure both images are converted to float32
    reconstructed_heatmap_pca = reconstructed_heatmap_pca.astype(np.float32)
    resized_img = resized_img.astype(np.float32)
                                     
    super_imposed_img_with_PCA = cv2.addWeighted(reconstructed_heatmap_pca, 0.5, resized_img.copy(), 0.5, 0)
    # If needed, convert back to uint8 for visualization
    super_imposed_img_with_PCA = super_imposed_img_with_PCA.astype(np.uint8)

    img_super_impose_name =str(idv_wsi_name)+'_image_heatmaps_with_PCA.jpg'
    final_img_super_imps_with_PCA_des = os.path.join(testing_results_wsi_output_,img_super_impose_name)
    cv2.imwrite(final_img_super_imps_with_PCA_des,super_imposed_img_with_PCA)



class FeatureHeatmapGenerator:


    def __init__(self, model_path, alpha=0.6, conv_name_heatmap='conv5_3_1x1_increase/bn'):
        model_4gpus = load_model(model_path)
        
        #pdb.set_trace()
        
        self.model = model_4gpus#.layers[-2]
        self.alpha = alpha
        self.conv_name_hm = conv_name_heatmap
        self.conv_name_fm = 'avg_pool'


    def heatmap_of_feature_from_dirs_save_same_dirs(self, scr_dir, feature_number):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
     
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)                         
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
                    if img_ext =='jpg':                  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            img = cv2.imread(image_path, cv2.IMREAD_COLOR)   
                            img_name_owext = img_name.split('.')[0]
                            img_name_4s = img_name_owext+'_heatmap.png'
                            cp_np_name = img_name_owext+'_class_prob.npy'
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            
                            # Generate heatmap and extract features and class probs.
                            fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
                            # save all representations...
                            final_dst_image = join_path(sub_dir_path, img_name)
                            final_dst_heatmap = join_path(sub_dir_path, img_name_4s)
                            final_class_prob_saving_path = join_path(sub_dir_path, cp_np_name)
                            final_fm_saving_path = join_path(sub_dir_path, fm_np_name)

                            cv2.imwrite(final_dst_image, img)
                            cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)


    def heatmap_of_feature_from_dirs_save_diff_dirs(self, scr_dir, feature_number, save_dir):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
     
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)      
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
                            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            img = cv2.resize(img, (1024,1024), interpolation = cv2.INTER_AREA)
                            img_name_owext = img_name.split('.')[0]
                            img_name_4s = img_name_owext+'_heatmap.png'
                            cp_np_name = img_name_owext+'_class_prob.npy'
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            
                            # Generate heatmap and extract features and class probs.
                            
                            fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
                            
                            #fam, conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                            # save all representations...
                            final_dst_image = join_path(dst_dir_final, img_name)
                            final_dst_heatmap = join_path(dst_dir_final, img_name_4s)
                            final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)
                            final_fm_saving_path = join_path(dst_dir_final, fm_np_name)

                            #cv2.imwrite(final_dst_image, img)
                            #cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)
                            

    def feature_extraction_class_probs_from_dirs_save_diff_dirs(self, scr_dir, feature_number, save_dir):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
        
        #pdb.set_trace()
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)      
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
                    #img_ext = img_name.split('.')[1]     
                    img_ext = img_name.split('.')[-1]  ## Only for testing case of LGG vs GBM.......
                    #pdb.set_trace()
                    if img_ext =='png':                  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            img = cv2.resize(img, (1024,1024), interpolation = cv2.INTER_AREA)
                            #img_name_owext = img_name.split('.')[0]   # General cases ..(all)
                            
                            img_name_owext = img_name.split('.')[0]+'.'+img_name.split('.')[1] ## Only for LGG s GBM testing .....
                            cp_np_name = img_name_owext+'_class_prob.npy'
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            
                            # Generate heatmap and extract features and class probs.
                            conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                            # save all representations...
                            final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)
                            final_fm_saving_path = join_path(dst_dir_final, fm_np_name)

                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)
                            
                            
    def feature_extract_class_prob_helper(self, image_path, feature_number):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)   
        
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        final_conv_layer_fm = self.model.get_layer(name=self.conv_name_fm)
        get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_fm.output])
        conv_output_fm = np.squeeze(get_output_fm([np.expand_dims(image / 255, axis=0)])[0]) 
        conv_output_fm = np.squeeze(conv_output_fm)
        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        #pdb.set_trace()
        dl_input = np.expand_dims(image / 255, axis=0)    
        class_probs = self.model.predict(dl_input)  
        
        return conv_output_fm, class_probs
    
    
    def feature_extract_class_prob_from_input1024x1024_image_helper(self, image, file_name, encoded_actual_labels_dir, output_dir):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''
        
        #image = cv2.imread(image_path)
        #image = cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)   
        
        #pdb.set_trace()
        
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        final_conv_layer_fm = self.model.get_layer(name=self.conv_name_fm)
        get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_fm.output])
        #conv_output_fm = np.squeeze(get_output_fm([np.expand_dims(image / 255, axis=0)])[0]) 
        conv_output_fm = np.squeeze(get_output_fm(image)[0]) 
        conv_output_fm = np.squeeze(conv_output_fm)
        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        #pdb.set_trace()
        #dl_input = np.expand_dims(image / 255, axis=0)    
        #dl_input = image
        class_probs = self.model.predict(image)  
        
        # Extract the 2D representation from ResNet50 model.....
        
        # generate heatmap
        
        final_conv_hm = self.model.get_layer(name=self.conv_name_hm)
        get_output_hm = K.function([self.model.layers[0].input], [final_conv_hm.output])
        conv_output_hm_2D_in = np.squeeze(get_output_hm(image)[0])   
        
        conv_output_hm_2D = conv_output_hm_2D_in[:, :, 16 - 1]
        conv_output_hm_2D /= np.max(conv_output_hm_2D)
        # >>>>>>>>>>>>>>>>>>>>>   mapping class called to the original labels......<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # 'GT_EC', 'GT_DEC', 'Group', 'Lineage', 'tumor_not_tumor
        
        
        #GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor.npy')
        #GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'GT_EC_GT_DEC_Group_Lineage_tvsnt_final.npy')
        GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'Copy_of_GT_EC_GT_DEC_Group_Lineage_tvsnt_final_bao_revised.npy')

        GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor = np.load((GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path),allow_pickle=True)
        
        #assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,1]))==74
        assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,2]))==4
        assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,3]))==6
        assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,4]))==2
        
        #encoded_file_path = join_path(encoded_actual_labels_dir,'74class_ResNet50_encoded_values_inorder.npy')
        #actual_label_file_path = join_path(encoded_actual_labels_dir,'74class_ResNet50_names_inorder.npy')
       

        
        y_encoded_values= GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,0]
        y_actual_name = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,1]
        
        y_encoded_values = np.array(y_encoded_values)
        y_actual_name = np.array(y_actual_name)
    
        
        max_index_col = np.argmax(class_probs, axis=1)
        result_idx = np.where(y_encoded_values == max_index_col[0])

        pred_class_name = y_actual_name[int(result_idx[0])]
        
        group_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),2]
        Lineage_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),3]
        Tumor_vs_non_tumor_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),4]
        
        print('Predicted as  :', pred_class_name)
        
        pred_class_cv = class_probs[0][int(max_index_col[0])]
        print('Predicted conf value  :', pred_class_cv)

        #y_pred_actual_names.append(pred_class_name)
        #y_pred_encoded_values.append(max_index_col)
        
        fv_file_path = join_path(output_dir,file_name+'_fv.npy')
        fv_2D_file_path = join_path(output_dir,file_name+'_fv_2D.npy')

        class_name_file_path = join_path(output_dir,file_name+'_cls_name.npy')
        class_conf_file_path = join_path(output_dir,file_name+'_conf_value.npy')
        
        group_pred_name_file_path = join_path(output_dir,file_name+'_group_name.npy')
        lineage_pred_file_path = join_path(output_dir,file_name+'_lineage_name.npy')
        Tumor_vs_non_tumor_file_path = join_path(output_dir,file_name+'_tvsnt_name.npy')
        
        np.save(fv_file_path,conv_output_fm)
        np.save(fv_2D_file_path,conv_output_hm_2D)

        np.save(class_name_file_path,pred_class_name)
        np.save(class_conf_file_path,class_probs)
        
        np.save(group_pred_name_file_path,group_pred)
        np.save(lineage_pred_file_path,Lineage_pred)
        np.save(Tumor_vs_non_tumor_file_path,Tumor_vs_non_tumor_pred)
        
        #return conv_output_fm, class_probs, pred_class_name, pred_class_cv
    
    def feature_extract_class_prob_from_input1024x1024_image_helper_heliyon_rebuttal(self, image, file_name, encoded_actual_labels_dir):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''
        
        #image = cv2.imread(image_path)
        #image = cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)   
        
        #pdb.set_trace()
        
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        final_conv_layer_fm = self.model.get_layer(name=self.conv_name_fm)
        get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_fm.output])
        conv_output_fm = np.squeeze(get_output_fm([np.expand_dims(image / 255, axis=0)])[0]) 
        #conv_output_fm = np.squeeze(get_output_fm(image)[0]) 
        conv_output_fm = np.squeeze(conv_output_fm)
        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        #pdb.set_trace()
        #dl_input = np.expand_dims(image / 255, axis=0)    
        #dl_input = image
        class_probs = self.model.predict(image)  
        
        # >>>>>>>>>>>>>>>>>>>>>   mapping class called to the original labels......<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # 'GT_EC', 'GT_DEC', 'Group', 'Lineage', 'tumor_not_tumor

        #GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor.npy')
        #GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'GT_EC_GT_DEC_Group_Lineage_tvsnt_final.npy')
        GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'Copy_of_GT_EC_GT_DEC_Group_Lineage_tvsnt_final_bao_revised.npy')

        GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor = np.load((GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path),allow_pickle=True)
        
        #assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,1]))==74
        assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,2]))==4
        assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,3]))==6
        assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,4]))==2
        
        #encoded_file_path = join_path(encoded_actual_labels_dir,'74class_ResNet50_encoded_values_inorder.npy')
        #actual_label_file_path = join_path(encoded_actual_labels_dir,'74class_ResNet50_names_inorder.npy')

        y_encoded_values= GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,0]
        y_actual_name = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,1]
        
        y_encoded_values = np.array(y_encoded_values)
        y_actual_name = np.array(y_actual_name)
    
        
        max_index_col = np.argmax(class_probs, axis=1)
        result_idx = np.where(y_encoded_values == max_index_col[0])

        pred_class_name = y_actual_name[int(result_idx[0])]
        
        group_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),2]
        Lineage_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),3]
        Tumor_vs_non_tumor_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),4]
        
        print('Predicted as  :', pred_class_name)
        
        pred_class_cv = class_probs[0][int(max_index_col[0])]
        print('Predicted conf value  :', pred_class_cv)

        #y_pred_actual_names.append(pred_class_name)
        #y_pred_encoded_values.append(max_index_col)
        
        #fv_file_path = join_path(output_dir,file_name+'_fv.npy')
        #class_name_file_path = join_path(output_dir,file_name+'_cls_name.npy')
        #class_conf_file_path = join_path(output_dir,file_name+'_conf_value.npy')
        
        #group_pred_name_file_path = join_path(output_dir,file_name+'_group_name.npy')
        #lineage_pred_file_path = join_path(output_dir,file_name+'_lineage_name.npy')
        #Tumor_vs_non_tumor_file_path = join_path(output_dir,file_name+'_tvsnt_name.npy')
        
        #np.save(fv_file_path,conv_output_fm)
        #np.save(class_name_file_path,pred_class_name)
        #np.save(class_conf_file_path,class_probs)
        
        #np.save(group_pred_name_file_path,group_pred)
        #np.save(lineage_pred_file_path,Lineage_pred)
        #np.save(Tumor_vs_non_tumor_file_path,Tumor_vs_non_tumor_pred)
        
        return conv_output_fm, class_probs, pred_class_name, group_pred,Lineage_pred,Tumor_vs_non_tumor_pred
    
    
    
    def feature_extract_class_prob_from_input1024x1024_image_with_heatmaps(self, image, feature_number, file_name, encoded_actual_labels_dir, output_dir):
         '''
         Returns the passed in image with a heatmap of the specific feature number overlayed
         :param model_path:
         :param image_path:
         :param feature_number:
         :param conv_name:
         :return:
         '''
         
         #image = cv2.imread(image_path)
         #image = cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)   
         
         #pdb.set_trace()
         
         # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
         final_conv_layer_fm = self.model.get_layer(name=self.conv_name_fm)
         get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_fm.output])
         #conv_output_fm = np.squeeze(get_output_fm([np.expand_dims(image / 255, axis=0)])[0]) 
         conv_output_fm = np.squeeze(get_output_fm(image)[0]) 
         conv_output_fm = np.squeeze(conv_output_fm)
         # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
   
         # generate heatmap
         
         final_conv_hm = self.model.get_layer(name=self.conv_name_hm)
         get_output_hm = K.function([self.model.layers[0].input], [final_conv_hm.output])
         conv_output = np.squeeze(get_output_hm(image)[0])    

         width, height, _ = image.shape  
         cam = conv_output[:, :, feature_number - 1]
         cam /= np.max(cam)
         cam_resized = cv2.resize(cam, (height, width))
         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
         # overlay heatmap on original image
         heatmap_final = cv2.addWeighted(image, self.alpha, heatmap, 1 - self.alpha, 0)
         
         img_super_impose_name_1 =str(file_name)+'_heatmaps.jpg'
         final_img_super_imps_des = os.path.join(output_dir,img_super_impose_name_1)
         cv2.imwrite(final_img_super_imps_des,heatmap_final)
         
         # >>>>>>>>>>>>>>>>. Apply PCA on top of CAM representation ........<<<<<<<<<<<<<<<<<<<<<<
         #CAM_heatmap_with_PCA(self, cam.copy(), image, file_name, output_dir) 
 
         reconstructed_image = []
             
         for i in range(3):  # Loop over the 3 color channels
                 image_2D = cam[:, :, i]
                 pca = PCA(n_components=min(image_2D.shape))  # Ensure valid n_components
                 transformed = pca.fit_transform(image_2D)  # Perform PCA
                 reconstructed_ = pca.inverse_transform(transformed)  # Reconstruct channel
                 reconstructed_image.append(reconstructed_)
             
         # Stack the reconstructed channels
         reconstructed_image = np.stack(reconstructed_image, axis=-1)
         reconstructed_heatmap_pca = 255*(np.clip(reconstructed_image, 0, 1))
             
         # Ensure both images are converted to float32
         reconstructed_heatmap_pca = reconstructed_heatmap_pca.astype(np.float32)
         #resized_img = resized_img.astype(np.float32)
                                          
         #super_imposed_img_with_PCA = cv2.addWeighted(reconstructed_heatmap_pca, 0.5, resized_img.copy(), 0.5, 0)
         super_imposed_img_with_PCA = cv2.addWeighted(image, self.alpha, reconstructed_heatmap_pca, 1 - self.alpha, 0)
         # If needed, convert back to uint8 for visualization
         super_imposed_img_with_PCA = super_imposed_img_with_PCA.astype(np.uint8)

         img_super_impose_name =str(file_name)+'_heatmaps_with_PCA.jpg'
         final_img_super_imps_with_PCA_des = os.path.join(output_dir,img_super_impose_name)
         cv2.imwrite(final_img_super_imps_with_PCA_des,super_imposed_img_with_PCA)
         
         # >>>>>>>>>  check the class probability.....
         class_probs = self.model.predict(image)  
         
         # >>>>>>>>>>>>>>>>>>>>>   mapping class called to the original labels......<<<<<<<<<<<<<<<<<<<<<<<<<
         
         # 'GT_EC', 'GT_DEC', 'Group', 'Lineage', 'tumor_not_tumor
         
         
         #GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor.npy')
         #GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'GT_EC_GT_DEC_Group_Lineage_tvsnt_final.npy')
         GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path = join_path(encoded_actual_labels_dir,'Copy_of_GT_EC_GT_DEC_Group_Lineage_tvsnt_final_bao_revised.npy')

         GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor = np.load((GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor_path),allow_pickle=True)
         
         #assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,1]))==74
         assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,2]))==4
         assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,3]))==6
         assert len(np.unique(GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,4]))==2
         
         #encoded_file_path = join_path(encoded_actual_labels_dir,'74class_ResNet50_encoded_values_inorder.npy')
         #actual_label_file_path = join_path(encoded_actual_labels_dir,'74class_ResNet50_names_inorder.npy')
        

         
         y_encoded_values= GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,0]
         y_actual_name = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[:,1]
         
         y_encoded_values = np.array(y_encoded_values)
         y_actual_name = np.array(y_actual_name)
     
         
         max_index_col = np.argmax(class_probs, axis=1)
         result_idx = np.where(y_encoded_values == max_index_col[0])

         pred_class_name = y_actual_name[int(result_idx[0])]
         
         group_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),2]
         Lineage_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),3]
         Tumor_vs_non_tumor_pred = GT_EC_GT_DEC_Group_Lineage_tumor_not_tumor[int(result_idx[0]),4]
         
         print('Predicted as  :', pred_class_name)
         
         pred_class_cv = class_probs[0][int(max_index_col[0])]
         print('Predicted conf value  :', pred_class_cv)

         #y_pred_actual_names.append(pred_class_name)
         #y_pred_encoded_values.append(max_index_col)
         
         fv_file_path = join_path(output_dir,file_name+'_fv.npy')
         class_name_file_path = join_path(output_dir,file_name+'_cls_name.npy')
         class_conf_file_path = join_path(output_dir,file_name+'_conf_value.npy')
         
         group_pred_name_file_path = join_path(output_dir,file_name+'_group_name.npy')
         lineage_pred_file_path = join_path(output_dir,file_name+'_lineage_name.npy')
         Tumor_vs_non_tumor_file_path = join_path(output_dir,file_name+'_tvsnt_name.npy')
         
         np.save(fv_file_path,conv_output_fm)
         np.save(class_name_file_path,pred_class_name)
         np.save(class_conf_file_path,class_probs)
         
         np.save(group_pred_name_file_path,group_pred)
         np.save(lineage_pred_file_path,Lineage_pred)
         np.save(Tumor_vs_non_tumor_file_path,Tumor_vs_non_tumor_pred)
         
    
    
    def generate_heatmaps_withPCA_input1024x1024(self, image, feature_number, file_name):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
             :param model_path:
             :param image_path:
             :param feature_number:
             :param conv_name:
             :return:
        '''
             
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<<
        
        #final_conv_layer_fm = self.model.get_layer(name=self.conv_name_fm)
        #get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_fm.output])
        #conv_output_fm = np.squeeze(get_output_fm([np.expand_dims(image / 255, axis=0)])[0]) 
        #conv_output_fm = np.squeeze(get_output_fm(image)[0]) 
        #conv_output_fm = np.squeeze(conv_output_fm)
        
        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
       
        # Generate heatmap    
        final_conv_hm = self.model.get_layer(name=self.conv_name_hm)
        get_output_hm = K.function([self.model.layers[0].input], [final_conv_hm.output])
        conv_output = np.squeeze(get_output_hm([np.expand_dims(image / 255, axis=0)])[0])    

        width, height, _ = image.shape  
        #cam = conv_output[:, :, feature_number - 1]
        cam = np.mean(conv_output, axis=2)
        cam /= np.max(cam)
        cam_resized = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # overlay heatmap on original image
        #image_uin8 = image.astype(uint8)
        heatmap = heatmap.astype(np.float32)

        heatmap_final = cv2.addWeighted(image, self.alpha, heatmap, 1 - self.alpha, 0)
        heatmap_final = heatmap_final.astype(np.uint8)

        #img_super_impose_name_1 =str(file_name)+'_heatmaps.jpg'
        #final_img_super_imps_des = os.path.join(output_dir,img_super_impose_name_1)
        #cv2.imwrite(final_img_super_imps_des,heatmap_final)
             
        # >>>>>>>>>>>>>>>>. Apply PCA on top of CAM representation ........<<<<<<<<<<<<<<<<<<<<<<
        #CAM_heatmap_with_PCA(self, cam.copy(), image, file_name, output_dir) 
     
        #cam_channels = np.expand_dims(cam,axis=-1)
        #matrix_rgb = np.stack([cam]*3, axis=-1)
        
        reconstructed_image = []
          
        #pdb.set_trace()
        
        # Loop over the 3 color channels 
        #for i in range(3):  
        #    image_2D = matrix_rgb[:, :, i]
        pca = PCA(n_components=min(cam.shape))  # Ensure valid n_components
        transformed = pca.fit_transform(cam)  # Perform PCA
        reconstructed_ = pca.inverse_transform(transformed)  # Reconstruct channel
        #reconstructed_image.append(reconstructed_)
        #reconstructed_ = reconstructed_.astype(np.float32)
        reconstructed_resized = cv2.resize(reconstructed_, (height, width))
                 
        # Stack the reconstructed channels
        #reconstructed_image = np.stack(reconstructed_image, axis=-1)
        #reconstructed_heatmap_pca = 255*(np.clip(reconstructed_image, 0, 1))
                 
        # Ensure both images are converted to float32
        heatmap_with_PCA = cv2.applyColorMap(np.uint8(255 * reconstructed_resized), cv2.COLORMAP_JET)
        heatmap_with_PCA = heatmap_with_PCA.astype(np.float32)

        #resized_img = resized_img.astype(np.float32)
                                              
        #super_imposed_img_with_PCA = cv2.addWeighted(reconstructed_heatmap_pca, 0.5, resized_img.copy(), 0.5, 0)
        super_imposed_img_with_PCA = cv2.addWeighted(image, self.alpha, heatmap_with_PCA, 1 - self.alpha, 0)
        # If needed, convert back to uint8 for visualization
        super_imposed_img_with_PCA = super_imposed_img_with_PCA.astype(np.uint8)

        #img_super_impose_name =str(file_name)+'_heatmaps_with_PCA.jpg'
        #final_img_super_imps_with_PCA_des = os.path.join(output_dir,img_super_impose_name)
        #cv2.imwrite(final_img_super_imps_with_PCA_des,super_imposed_img_with_PCA)
        
        return heatmap_final,  super_imposed_img_with_PCA
    
         
    def _heatmap_of_feature_helper(self, image_path, feature_number):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
        width, height, _ = image.shape        
        #for layer in self.model.layers:
        #   print(layer.output_shape)
         #  print(layer.name)
        final_conv_layer = self.model.get_layer(name=self.conv_name_hm)
        get_output = K.function([self.model.layers[0].input], [final_conv_layer.output])
        conv_output = np.squeeze(get_output([np.expand_dims(image / 255, axis=0)])[0])    

        #h,w, feature_number =   conv_output.shape        
        # generate heatmap
        cam = conv_output[:, :, feature_number - 1]
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image
        heatmap_final = cv2.addWeighted(image, self.alpha, heatmap, 1 - self.alpha, 0)
        # xxxxxxxxx  Heatmap section is done here   xxxxxxxxxxxxxx     
        
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        final_conv_layer_fm = self.model.get_layer(name=self.conv_name_fm)
        get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_fm.output])
        conv_output_fm = np.squeeze(get_output_fm([np.expand_dims(image / 255, axis=0)])[0]) 
        conv_output_fm = np.squeeze(conv_output_fm)
        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        #pdb.set_trace()
        dl_input = np.expand_dims(image / 255, axis=0)    
        class_probs = self.model.predict(dl_input)  
        
        return heatmap_final , conv_output_fm, class_probs
    
    def read_numpy_files_for_patches_of_WSI(self, numpy_file_path):
    
        image_npy = [x for x in sorted(os.listdir(numpy_file_path)) if x[-7:] == '_fv.npy']
        
        WSI_ID = []
        patch_IDS = []
        fv_all = []
        class_name_all = []
        cfv_all = []
        
        #pdb.set_trace()    
        
        for i, f in enumerate(image_npy):
            #ext = os.path.splitext(f)[1]
            #if ext.lower() not in valid_images:
            #    continue
            
            file_name,file_ext = os.path.splitext(f)  
    
            actual_file_name = file_name.split('_fv')[0]   
            patch_IDS.append(actual_file_name)
            print('Workig for ',actual_file_name)
            
            if i ==0:
                WSI_ID = actual_file_name.split('_')[0]
                   
            fv_file_name = file_name+'.npy'
            class_file_name = actual_file_name+'_cls_name.npy'
            confv_file_name = actual_file_name+'_conf_value.npy'
            
            fv_sub_dir = join_path(numpy_file_path,fv_file_name)
            class_sub_dir = join_path(numpy_file_path,class_file_name)
            cfv_sub_dir = join_path(numpy_file_path,confv_file_name)
            
            fv_file = np.load(fv_sub_dir)
            class_file = np.load(class_sub_dir)
            cfv_file = np.load(cfv_sub_dir)
            class_name_all.append(class_file)
            
            fv_file = np.expand_dims(fv_file,axis=0)
            #fv_file = np.expand_dims(fv_file,axis=0)
           
            if i==0:
                fv_all = fv_file
                #class_name_all = class_file
                cfv_all = cfv_file
            else:
                fv_all = np.concatenate((fv_all, fv_file),axis =0)
                #class_name_all = np.concatenate((class_name_all, class_file),axis =1)
                cfv_all = np.concatenate((cfv_all,cfv_file),axis =0)
        
        #pdb.set_trace()
        
        patch_IDS = np.array(patch_IDS)
        fv_all = np.array(fv_all)
        class_name_all = np.array(class_name_all)
        cfv_all = np.array(cfv_all)
            
        np.save(join_path(numpy_file_path,WSI_ID+'_IDs_all.npy'),patch_IDS)        
        np.save(join_path(numpy_file_path,WSI_ID+'_fvs_all.npy'),fv_all) 
        np.save(join_path(numpy_file_path,WSI_ID+'_names_all.npy'),class_name_all)  
        np.save(join_path(numpy_file_path,WSI_ID+'_confvs_all.npy'),cfv_all)  
        
        #return patch_IDS,fv_all,class_name_all,cfv_all

    
    def create_heatmap(self, im_map, im_cloud, kernel_size=(3,3),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
        '''
        img is numpy array
        kernel_size must be odd ie. (5,5)
        ''' 
        # create blur image, kernel must be an odd number
        im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)
        im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)
        return (a1*im_map + a2*im_cloud_clr).astype(np.uint8) 
    
    
    def extract_same_size_patches_from_wsi_with_xy_final_PIL(self, svs_img_dir, patches_saving_dir, patch_size):
        
    #    patch_dir_name = 'patches_'+str(patch_size[0])+'/ 
    #    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
    #        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
    #    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    #    
        patches_dir = patches_saving_dir
        
        image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.tif' or '.svs']
        
        #pdb.set_trace()    
        #for f in os.listdir(image_svs):
        for i, f in enumerate(image_svs):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            
            dir_name = os.path.splitext(f)[0]              
            print(svs_img_dir.split('/')[0])        
            if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
                os.makedirs("%s/%s"%(patches_dir,dir_name))
            
            patches_sub_dir = join_path(patches_dir,dir_name+'/')
            # open scan
            svs_img_path = os.path.join(svs_img_dir,f)
            scan = openslide.OpenSlide(svs_img_path)
            
            scan_dimensions = scan.dimensions        
            orig_w = scan_dimensions[0]
            orig_h = scan_dimensions[1]           
            no_patches_x_axis = orig_w/patch_size[0]
            no_patches_y_axis = orig_h/patch_size[1]                

            starting_row_columns = []
            img_saving_idx = 0        

            for y in range(0,orig_h,patch_size[1]):
                for x in range(0, orig_w,patch_size[0]):                
                    # save only those HPF patches that satify the following condition...
                    if x+patch_size[0] <= orig_w and y+patch_size[1] <= orig_h:
                        img = np.array(scan.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]                   
                    
                    idx_sr_sc = str(img_saving_idx)+'_'+str(x)+'_'+str(y)                
                    starting_row_columns.append(idx_sr_sc)
                    print("Processing:"+str(img_saving_idx))                
                    ac_img_name =dir_name+'_'+str(img_saving_idx)+'_'+str(x)+'_'+str(y)+'.jpg'
                    final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                    
                    im_PIL = Image.fromarray(img)
                    im_PIL.save(final_img_des)
                    #cv2.imwrite(final_img_des,img)
                    #imsave(final_img_des,img)                
                    img_saving_idx +=1
                    
            scan.close 
        
            #pdb.set_trace()
             
            svs_log = {}
            svs_log["ID"] = dir_name
            svs_log["height"] = orig_h
            svs_log["width"] = orig_w
            svs_log["patch_width"] = patch_size[0]
            svs_log["patch_height"] = patch_size[1]
            svs_log["no_patches_x_axis"] = no_patches_x_axis
            svs_log["no_patches_y_axis"] = no_patches_y_axis
            svs_log["number_HPFs_patches"] = img_saving_idx
            svs_log["starting_rows_columns"] = starting_row_columns
             
            # make experimental log saving path...
            json_file = os.path.join(patches_sub_dir,'image_patching_log.json')
            with open(json_file, 'w') as file_path:
                json.dump(svs_log, file_path, indent=4, sort_keys=True)
            
            
        return patches_sub_dir




def generate_heatmaps_withPCA(image, hatmap_cam, patch_size):
    '''
     Returns the passed in image with a heatmap of the specific feature number overlayed
          :param model_path:
          :param image_path:
          :param feature_number:
          :param conv_name:
          :return:
    '''
    
    alpha = 0.6    
     # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<<

    cam_resized = cv2.resize(hatmap_cam, (patch_size[0], patch_size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
     
    # overlay heatmap on original image
    heatmap = heatmap.astype(np.float32)
    heatmap_final = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    heatmap_final = heatmap_final.astype(np.uint8)
     
    # >>>>>>>>>>>>>>>>. Apply PCA on top of CAM representation ........<<<<<<<<<<<<<<<<<<<<<<   
    #reconstructed_image = []
    pca = PCA(n_components=min(hatmap_cam.shape))  # Ensure valid n_components
    transformed = pca.fit_transform(hatmap_cam)  # Perform PCA
    reconstructed_ = pca.inverse_transform(transformed)  # Reconstruct channel
    reconstructed_resized = cv2.resize(reconstructed_, ((patch_size[0], patch_size[1])))
    heatmap_with_PCA = cv2.applyColorMap(np.uint8(255 * reconstructed_resized), cv2.COLORMAP_JET)
    heatmap_with_PCA = heatmap_with_PCA.astype(np.float32)

    super_imposed_img_with_PCA = cv2.addWeighted(image, alpha, heatmap_with_PCA, 1 - alpha, 0)
    super_imposed_img_with_PCA = super_imposed_img_with_PCA.astype(np.uint8)

    return heatmap_final,  super_imposed_img_with_PCA
 

def read_numpy_files_for_patches_of_WSI_for_74_classes(numpy_file_path,merged_results_saving_path):
      
    #pdb.set_trace()
    
    image_npy = [x for x in sorted(os.listdir(numpy_file_path)) if x[-7:] == '_fv.npy']
    
    WSI_ID = []
    patch_IDS = []
    fv_all = []
    fv_2D_all = []

    class_name_all = []
    cfv_all = []
    
    grp_name_all = []
    lng_name_all  = []
    tvsnt_name_all = []
    
    #pdb.set_trace()    
    
    for i, f in enumerate(image_npy):
        #ext = os.path.splitext(f)[1]
        #if ext.lower() not in valid_images:
        #    continue
        
        file_name,file_ext = os.path.splitext(f)  

        actual_file_name = file_name.split('_fv')[0]   
        patch_IDS.append(actual_file_name)
        print('Workig for ',actual_file_name)
        
        if i ==0:
            WSI_ID = actual_file_name.split('_')[0]
               
        fv_file_name = actual_file_name+'_fv.npy'
        fv_2D_file_name = actual_file_name+'_fv_2D.npy'
        class_file_name = actual_file_name+'_cls_name.npy'
        confv_file_name = actual_file_name+'_conf_value.npy'
        
        grp_file_name = actual_file_name+'_group_name.npy'
        lng_file_name =  actual_file_name+'_lineage_name.npy'
        tvnt_file_name = actual_file_name+'_tvsnt_name.npy'
        
        fv_sub_dir = join_path(numpy_file_path,fv_file_name)
        fv_2D_sub_dir = join_path(numpy_file_path,fv_2D_file_name)

        class_sub_dir = join_path(numpy_file_path,class_file_name)
        cfv_sub_dir = join_path(numpy_file_path,confv_file_name)
        
        grp_sub_dir = join_path(numpy_file_path,grp_file_name)
        lng_sub_dir = join_path(numpy_file_path,lng_file_name)
        tvsnt_sub_dir = join_path(numpy_file_path,tvnt_file_name)
        
        
        fv_file = np.load(fv_sub_dir)
        fv_2D_file = np.load(fv_2D_sub_dir)
        exd_fv_2D_file = np.expand_dims(fv_2D_file,axis = 0)

        class_file = np.load(class_sub_dir)
        cfv_file = np.load(cfv_sub_dir)
        class_name_all.append(class_file)
        
        grp_call = np.load(grp_sub_dir)
        lng_call = np.load(lng_sub_dir)
        tvnt_call = np.load(tvsnt_sub_dir)
        
        grp_name_all.append(grp_call)
        lng_name_all.append(lng_call)
        tvsnt_name_all.append(tvnt_call)
        
        
        fv_file = np.expand_dims(fv_file,axis=0)
        #fv_file = np.expand_dims(fv_file,axis=0)
       
        if i==0:
            fv_all = fv_file
            fv_2D_all = exd_fv_2D_file
            cfv_all = cfv_file
        else:
            fv_all = np.concatenate((fv_all, fv_file),axis =0)
            cfv_all = np.concatenate((cfv_all,cfv_file),axis =0)
            fv_2D_all = np.concatenate((fv_2D_all,exd_fv_2D_file),axis=0)
    
    #pdb.set_trace()
    
    patch_IDS = np.array(patch_IDS)
    fv_all = np.array(fv_all)
    fv_2D_all = np.array(fv_2D_all)
    class_name_all = np.array(class_name_all)
    cfv_all = np.array(cfv_all)
    
    grp_name_all = np.array(grp_name_all)
    lng_name_all = np.array(lng_name_all)
    tvsnt_name_all = np.array(tvsnt_name_all)
    
    #assert patch_IDS.shape[0]==
    np.save(join_path(merged_results_saving_path,WSI_ID+'_IDs_all.npy'),patch_IDS)        
    np.save(join_path(merged_results_saving_path,WSI_ID+'_fvs_all.npy'),fv_all) 
    np.save(join_path(merged_results_saving_path,WSI_ID+'_fvs_2D_all.npy'),fv_2D_all) 

    np.save(join_path(merged_results_saving_path,WSI_ID+'_names_all.npy'),class_name_all)  
    np.save(join_path(merged_results_saving_path,WSI_ID+'_confvs_all.npy'),cfv_all) 
    
    np.save(join_path(merged_results_saving_path,WSI_ID+'_group_all.npy'),grp_name_all)  
    np.save(join_path(merged_results_saving_path,WSI_ID+'_lineage_all.npy'),lng_name_all)  
    np.save(join_path(merged_results_saving_path,WSI_ID+'_tumor_vs_not_tumor_all.npy'),tvsnt_name_all)  


  

def testing_idv_WSI_from_patch_dir_merging_outputs(trained_model_obj, patches_dir, encoded_actual_labels_dir, testing_log_saving_path_numpy_files):

    
    #pdb.set_trace()
    # Load the testing data from data directory...
    print("Loading image from the directorty ...")              
    
    #images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png' or '.svs' or '.tif']
    #images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png']
    images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    print('Total number of images:'+str(len(images_name)))
    Total_steatosis_cell_wsi = 0
    Total_pixels = 0
    total_segmentated_pixels_wsi = 0
    
    #pdb.set_trace()
    
    for i, img_name in enumerate(images_name):
        
        ext = os.path.splitext(img_name)[1]    
        img_name_wo_ext = os.path.splitext(img_name)[0]
        img_path = os.path.join(patches_dir,img_name)
        #img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)#.astype("int16").astype('float32')
        
        img_ext = ext.split('.')[-1]  ## Only for testing case of LGG vs GBM.......
        print('Input extension is :', img_ext)
        #pdb.set_trace()
        if img_ext =='jpg' or 'png': 
            
            #img = read_and_preprocess_img(img_path, size=(1024,1024))
            img = read_and_wo_preprocess_img(img_path, size=(1024,1024))       
            #gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            trained_model_obj.feature_extract_class_prob_from_input1024x1024_image_helper(img, img_name_wo_ext, encoded_actual_labels_dir, testing_log_saving_path_numpy_files)
            #samples_ids_train, x_train,y_ac_train, conf_values = fhg.read_numpy_files_for_patches_of_WSI(testing_log_saving_path_numpy_files)
        
    # Call a function to merge all files..... generate final log for indivisual WSI
    
    #read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files)  
    

def testing_idv_WSI_from_patch_dir_merging_outputs_rebuttal_heliyon(trained_model_obj, patches_dir, encoded_actual_labels_dir, testing_log_saving_path_numpy_files):

    
    #pdb.set_trace()
    # Load the testing data from data directory...
    print("Loading image from the directorty ...")              
    
    #images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png' or '.svs' or '.tif']
    #images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png']
    images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    print('Total number of images:'+str(len(images_name)))
    Total_steatosis_cell_wsi = 0
    Total_pixels = 0
    total_segmentated_pixels_wsi = 0
    
    #pdb.set_trace()
    
    for i, img_name in enumerate(images_name):
        
        ext = os.path.splitext(img_name)[1]    
        img_name_wo_ext = os.path.splitext(img_name)[0]
        img_path = os.path.join(patches_dir,img_name)
        #img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)#.astype("int16").astype('float32')
        
        img_ext = ext.split('.')[-1]  ## Only for testing case of LGG vs GBM.......
        print('Input extension is :', img_ext)
        #pdb.set_trace()
        if img_ext =='jpg' or 'png': 
            
            #img = read_and_preprocess_img(img_path, size=(1024,1024))
            img = read_and_wo_preprocess_img(img_path, size=(1024,1024))       
            #gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            trained_model_obj.feature_extract_class_prob_from_input1024x1024_image_with_heatmaps(img, img_name_wo_ext, encoded_actual_labels_dir, testing_log_saving_path_numpy_files)
            #samples_ids_train, x_train,y_ac_train, conf_values = fhg.read_numpy_files_for_patches_of_WSI(testing_log_saving_path_numpy_files)
        
    # Call a function to merge all files..... generate final log for indivisual WSI
    
    #read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files)  
    
    
def testing_WSIs_from_the_patch_dir(trained_model_path, encoded_actual_labels_dir, WSIs_and_patch_dir, wsi_filenames, testing_log_saving_path):
    
    #pdb.set_trace()
    
    # load weights to model
    print('-'*30)
    print('Loading model ...')
    print('-'*30)
    
    # Testing with the final model.......
    best_model_name = [x for x in sorted(os.listdir(trained_model_path)) if x[-14:] == '_best_model.h5']
    
    # Testing with the final model.......
    #best_model_name = ['final_model.h5']
    
    model_path = os.path.join(trained_model_path,best_model_name[0])
    trained_model_obj = FeatureHeatmapGenerator(model_path)
    
    
    # Process all WSI image available ...... keep the directory information...
    
    testing_dir_all = []
        
    for wsi_filename in wsi_filenames:

        #slide_name = os.path.splitext(wsi_filename)[0]
        slide_name = wsi_filename
        patches_dir = os.path.join(WSIs_and_patch_dir, slide_name+'_patches')

        #pdb.set_trace()
        
        if not os.path.isdir("%s/%s"%(testing_log_saving_path,slide_name+'_outputs')):
            os.makedirs("%s/%s"%(testing_log_saving_path,slide_name+'_outputs'))        

        testing_log_saving_path_2 = join_path(testing_log_saving_path,slide_name+'_outputs')


        if not os.path.isdir("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/')):
            os.makedirs("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/'))   
        testing_log_saving_path_numpy_files = join_path(testing_log_saving_path_2,'outputs_numpy_logs/') 
        
        # Generate the class level call for each patches ... for idv WSI..........
        testing_idv_WSI_from_patch_dir_merging_outputs(trained_model_obj, patches_dir, encoded_actual_labels_dir, testing_log_saving_path_numpy_files)
        
        merged_results_saving_path = testing_log_saving_path_2
        read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files, merged_results_saving_path) 
        
        testing_dir_all.append(testing_log_saving_path_2)
    
    # Take the path ... for future use .... 
    testing_outputs_dir_all = np.array(testing_dir_all)
    class_logs_directory_name = 'outputs_numpy_logs'    
        
    return testing_outputs_dir_all, class_logs_directory_name 
       

def testing_WSIs_from_the_patch_dir_rebuttal_heliyon(trained_model_path, encoded_actual_labels_dir, WSIs_and_patch_dir, wsi_filenames, testing_log_saving_path):
    
    #pdb.set_trace()
    
    # load weights to model
    print('-'*30)
    print('Loading model ...')
    print('-'*30)
    
    # Testing with the final model.......
    best_model_name = [x for x in sorted(os.listdir(trained_model_path)) if x[-14:] == '_best_model.h5']
    
    # Testing with the final model.......
    #best_model_name = ['final_model.h5']
    
    model_path = os.path.join(trained_model_path,best_model_name[0])
    trained_model_obj = FeatureHeatmapGenerator(model_path)
    
    
    # Process all WSI image available ...... keep the directory information...
    
    testing_dir_all = []
        
    for wsi_filename in wsi_filenames:

        #slide_name = os.path.splitext(wsi_filename)[0]
        slide_name = wsi_filename
        patches_dir = os.path.join(WSIs_and_patch_dir, slide_name)

        #pdb.set_trace()
        
        if not os.path.isdir("%s/%s"%(testing_log_saving_path,slide_name+'_outputs')):
            os.makedirs("%s/%s"%(testing_log_saving_path,slide_name+'_outputs'))        

        testing_log_saving_path_2 = join_path(testing_log_saving_path,slide_name+'_outputs')


        if not os.path.isdir("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/')):
            os.makedirs("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/'))   
        testing_log_saving_path_numpy_files = join_path(testing_log_saving_path_2,'outputs_numpy_logs/') 
        
        # Generate the class level call for each patches ... for idv WSI..........
        testing_idv_WSI_from_patch_dir_merging_outputs_rebuttal_heliyon(trained_model_obj, patches_dir, encoded_actual_labels_dir, testing_log_saving_path_numpy_files)
        
        merged_results_saving_path = testing_log_saving_path_2
        read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files, merged_results_saving_path) 
        
        testing_dir_all.append(testing_log_saving_path_2)
    
    # Take the path ... for future use .... 
    testing_outputs_dir_all = np.array(testing_dir_all)
    class_logs_directory_name = 'outputs_numpy_logs'    
        
    return testing_outputs_dir_all, class_logs_directory_name 


def testing_KDB_from_the_patch_dir(trained_model_path, encoded_actual_labels_dir, WSIs_and_patch_dir, wsi_filenames, testing_log_saving_path):
    
    #pdb.set_trace()
    
    # load weights to model
    print('-'*30)
    print('Loading model ...')
    print('-'*30)
    
    # Testing with the final model.......
    best_model_name = [x for x in sorted(os.listdir(trained_model_path)) if x[-14:] == '_best_model.h5']
    
    # Testing with the final model.......
    #best_model_name = ['final_model.h5']
    
    model_path = os.path.join(trained_model_path,best_model_name[0])
    trained_model_obj = FeatureHeatmapGenerator(model_path)
    
    
    # Process all WSI image available ...... keep the directory information...
    
    testing_dir_all = []
        
    for wsi_filename in wsi_filenames:

        #slide_name = os.path.splitext(wsi_filename)[0]
        slide_name = wsi_filename
        patches_dir = os.path.join(WSIs_and_patch_dir, slide_name)

        #pdb.set_trace()
        
        if not os.path.isdir("%s/%s"%(testing_log_saving_path,slide_name+'_outputs')):
            os.makedirs("%s/%s"%(testing_log_saving_path,slide_name+'_outputs'))        

        testing_log_saving_path_2 = join_path(testing_log_saving_path,slide_name+'_outputs')


        if not os.path.isdir("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/')):
            os.makedirs("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/'))   
        testing_log_saving_path_numpy_files = join_path(testing_log_saving_path_2,'outputs_numpy_logs/') 
        
        # Generate the class level call for each patches ... for idv WSI..........
        testing_idv_WSI_from_patch_dir_merging_outputs(trained_model_obj, patches_dir, encoded_actual_labels_dir, testing_log_saving_path_numpy_files)
        
        #read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files) 
        merged_results_saving_path = testing_log_saving_path_2
        read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files, merged_results_saving_path) 
        
        testing_dir_all.append(testing_log_saving_path_2)
    
    # Take the path ... for future use .... 
    testing_outputs_dir_all = np.array(testing_dir_all)
    class_directory_name = 'outputs_numpy_logs'    
        
    return testing_outputs_dir_all, class_directory_name 

       
def testing_LCDB_from_the_patch_dir(trained_model_path, encoded_actual_labels_dir, WSIs_and_patch_dir, wsi_filenames, testing_log_saving_path):
    
    #pdb.set_trace()
    
    # load weights to model
    print('-'*30)
    print('Loading model ...')
    print('-'*30)
    
    # Testing with the final model.......
    best_model_name = [x for x in sorted(os.listdir(trained_model_path)) if x[-14:] == '_best_model.h5']
    
    # Testing with the final model.......
    #best_model_name = ['final_model.h5']
    
    model_path = os.path.join(trained_model_path,best_model_name[0])
    trained_model_obj = FeatureHeatmapGenerator(model_path)
    
    
    # Process all WSI image available ...... keep the directory information...
    
    testing_dir_all = []
        
    for wsi_filename in wsi_filenames:

        #slide_name = os.path.splitext(wsi_filename)[0]
        slide_name = wsi_filename
        patches_dir = os.path.join(WSIs_and_patch_dir, slide_name)

        #pdb.set_trace()
        
        if not os.path.isdir("%s/%s"%(testing_log_saving_path,slide_name+'_outputs')):
            os.makedirs("%s/%s"%(testing_log_saving_path,slide_name+'_outputs'))        

        testing_log_saving_path_2 = join_path(testing_log_saving_path,slide_name+'_outputs')


        if not os.path.isdir("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/')):
            os.makedirs("%s/%s"%(testing_log_saving_path_2,'outputs_numpy_logs/'))   
        testing_log_saving_path_numpy_files = join_path(testing_log_saving_path_2,'outputs_numpy_logs/') 
        
        # Generate the class level call for each patches ... for idv WSI..........
        testing_idv_WSI_from_patch_dir_merging_outputs(trained_model_obj, patches_dir, encoded_actual_labels_dir, testing_log_saving_path_numpy_files)
        
        #read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files) 
        
        merged_results_saving_path = testing_log_saving_path_2
        read_numpy_files_for_patches_of_WSI_for_74_classes(testing_log_saving_path_numpy_files, merged_results_saving_path) 
        
        testing_dir_all.append(testing_log_saving_path_2)
    
    # Take the path ... for future use .... 
    testing_outputs_dir_all = np.array(testing_dir_all)
    class_directory_name = 'outputs_numpy_logs'    
        
    return testing_outputs_dir_all, class_directory_name 
