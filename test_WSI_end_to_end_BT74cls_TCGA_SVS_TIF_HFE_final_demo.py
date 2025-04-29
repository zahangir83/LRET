#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:46:15 2024

@author: malom
"""

import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from os.path import join as join_path
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sn
import shutil
import json
from PIL import Image
import scipy.ndimage as ndimage
kernel = np.ones((3,3), np.uint8) 
font = cv2.FONT_HERSHEY_SIMPLEX
from wsi_utils import wsi_inference as wsi_inf
from wsi_utils import wsi_utils_functions
import pandas as pd

import pdb

 
parser = argparse.ArgumentParser()
parser.add_argument('--project_name_classification', type=str, default="74CLS_Path_ResNet50_2GPUs_final", help='Name of your project')
parser.add_argument('--HPF_height', type=int, default=1024, help='Height of cropped input image to network') #1152
parser.add_argument('--HPF_width', type=int, default=1024, help='Width of cropped input image to network')
#parser.add_argument('--seg_height', type=int, default=256, help='Height of cropped input image to network')
# parser.add_argument('--seg_width', type=int, default=256, help='Width of cropped input image to network')
# parser.add_argument('--clas_height', type=int, default=256, help='Height of cropped input image to network')
# parser.add_argument('--clas_width', type=int, default=256, help='Width of cropped input image to network')
# parser.add_argument('--model_seg', type=str, default="R2UNet", help='The model you are using. See model_builder.py for supported models')

# Directory for Actual labels and encoded values....
parser.add_argument('--encoded_actual_labels_dir', type=str, default="/research/rgs01/home/clusterHome/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/74_clas_DB/Class_group_lineage/", help='Dataset you are using.')
# Directory for LGG
#parser.add_argument('--input_svs_image_path', type=str, default="/home/malom/stjude_projects/computational_pathology/74_classes_project/project_74_classes_digital_path/database/testing_dataset/WSI_TCGA_LGG_GBM/LGG_GBM/LGG/", help='Dataset you are using.')
# Directory for GBM....
#parser.add_argument('--input_svs_image_path', type=str, default="/home/malom/stjude_projects/computational_pathology/74_classes_project/project_74_classes_digital_path/database/testing_dataset/WSI_TCGA_LGG_GBM/LGG_GBM/GBM/", help='Dataset you are using.')

# >>>>>>>>>>>>>>>>>>>>>>>>>> GBM_TCIA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
parser.add_argument('--input_svs_image_path', type=str, default="/home/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/74_clas_DB/TCGA_GBM_LGG_examples/TCGA_GBM_LGG_SVS_files/", help='Dataset you are using.')


args = parser.parse_args()

print("creating all the necessary diretoreis:")
# project_log_path_seg = join_path('experimental_logs/', args.project_name_segmentation+'/')   
# training_log_saving_path_seg = join_path(project_log_path_seg,'training/')
# testing_log_saving_path_seg_1 = join_path(project_log_path_seg,'testing/')
# weight_loading_path_seg = join_path(project_log_path_seg,'weights/')

project_log_path_clas = join_path('experimental_logs/', args.project_name_classification+'/')   
training_log_saving_path_clas = join_path(project_log_path_clas,'training/')
testing_log_saving_path_clas = join_path(project_log_path_clas,'testing/')


proj_outputs_name_final = (args.input_svs_image_path).split('/')[-2]+'_ouptuts'

if not os.path.isdir("%s/%s"%(testing_log_saving_path_clas,proj_outputs_name_final)):
    os.makedirs("%s/%s"%(testing_log_saving_path_clas,proj_outputs_name_final))        

testing_log_saving_path_final = join_path(testing_log_saving_path_clas,proj_outputs_name_final)


trained_model_path = join_path(project_log_path_clas,'trained_model/')
#weight_loading_path_clas = join_path(project_log_path_clas,'weights/')
# create pathces from the WSI
HPFs_size = (args.HPF_height,args.HPF_width)  
# Step_1 : >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    Extract specific size of the patches       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#pdb.set_trace()
directory_names_all = wsi_utils_functions.extract_same_size_patches_from_wsi_with_xy_PIL_SVS_TIF(args.input_svs_image_path, HPFs_size, testing_log_saving_path_final) # Using PIL ......
# directory_names_all = ['C3L-00016-21']

# Step_2 : >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    Testing the model.       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#pdb.set_trace()
WSIs_and_patche_dir = testing_log_saving_path_final
#directory_names_all = ['a1980574-357f-11eb-8cd0-001a7dda7111']
testing_outputs_dir_all, class_logs_directory_name  = wsi_inf.testing_WSIs_from_the_patch_dir(trained_model_path, args.encoded_actual_labels_dir, WSIs_and_patche_dir, directory_names_all, testing_log_saving_path_final)

######## testing_outputs_dir_all = ['experimental_logs/74CLS_Path_ResNet50_2GPUs_final/testing/GBM_TCIA_idv_test_ouptuts/C3L-00016-21_outputs']
######## class_logs_directory_name = 'outputs_numpy_logs'

#Step 3 : >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Generate WSI level ouptuts   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< .... 
# >>>>>>>>>>>>>>>>>>>>>>.. CREATING distionary from the directory name manually <<<<<<<<<<<<<<<<<<<<<<<<<<<<,
# image_svs_names = [x for x in sorted(os.listdir(args.input_svs_image_path)) if x[-4:] =='.svs']
# directory_names_all = []

# for i, f in enumerate(image_svs_names):          
#     dir_name = os.path.splitext(f)[0] 
#     directory_names_all.append(dir_name) 

# directory_names_all = np.array(directory_names_all)        

#pdb.set_trace()
# load the encoded labels file ....
df_pred_encoded_actul_group_lineage_tvsnt = pd.read_csv(os.path.join(args.encoded_actual_labels_dir,'Copy_of_GT_EC_GT_DEC_Group_Lineage_tvsnt_final_bao_revised.csv')) 
df_sorted_pred_encoded_actul_group_lineage_tvsnt = df_pred_encoded_actul_group_lineage_tvsnt.sort_values('GT_EC')
sorted_labels = df_sorted_pred_encoded_actul_group_lineage_tvsnt['GT_DEC']

Threshold = 0.9
num_wsi = len(directory_names_all)

counts_for_all_cases = np. zeros((num_wsi, len(sorted_labels))) 

caseID_final = []

flag_break = 0

for kk in range(num_wsi):
    
    #if directory_names_all[kk] !='C3L-04213-25':
    print('Working for the case :', directory_names_all[kk])     
        
    WSIs_and_patche_dir_final = os.path.join(WSIs_and_patche_dir,directory_names_all[kk])
    merged_testing_log_saving_path_final = os.path.join(WSIs_and_patche_dir,directory_names_all[kk]+'_outputs')
    #wsi_utils_functions.patches_to_image_from_prediction_draw_ROI_WSI_for_BT74class_final( WSIs_and_patche_dir_final, testing_log_saving_path_final)
    df_matrix_unquelb_counts_indv_case = wsi_utils_functions.patches_to_image_from_prediction_draw_ROI_WSI_for_BT74class_final_cv2( WSIs_and_patche_dir_final, Threshold, merged_testing_log_saving_path_final)
        #pdb.set_trace()
        #wsi_utils_functions.bar_plot_from_the_model_prediction_and_count(WSIs_and_patche_dir_final,directory_names_all[kk],0.9)
    
    pred_labels = df_matrix_unquelb_counts_indv_case['Pred_Labels']
    pred_labels = np.array(pred_labels)
    counts_for_labels = df_matrix_unquelb_counts_indv_case['Count']
    counts_for_labels = np.array(counts_for_labels)
            
    for jj in range(len(pred_labels)):
            
        indx_for_idv_labels = np.where(sorted_labels == pred_labels[jj])
        counts_for_all_cases[kk,int(indx_for_idv_labels[0])] = counts_for_labels[jj]
            
        
    caseID_final.append(directory_names_all[kk])
        
    print('Patient count :', kk)
    
    #if kk == 213:
    #    break
   
   

pdb.set_trace()

counts_for_all_cases = np.array(counts_for_all_cases)
caseID_final = np.array(caseID_final)

counts_for_all_cases_for_N_case = counts_for_all_cases[:len(caseID_final),:]
df_outputs_final = pd.DataFrame(counts_for_all_cases, index = caseID_final, columns = sorted_labels )
df_outputs_final = pd.DataFrame(counts_for_all_cases_for_N_case, index = caseID_final, columns = sorted_labels )

#testing_log_saving_path_final

df_outputs_final.to_csv(os.path.join(testing_log_saving_path_final,proj_outputs_name_final+'_'+str(Threshold)+'_73'+'.csv'))

# if not os.path.isdir("%s/%s"%(base_dir,'output_images/')):
#     os.makedirs("%s/%s"%(base_dir,'output_images/'))   
# testing_log_saving_path_for_images = join_path(base_dir,'output_images/') 

# wsi_class_obj.patches_to_image_from_prediction_for_BT74class_final(patches_dir,classifier_logs, outputs_saving_dir)



