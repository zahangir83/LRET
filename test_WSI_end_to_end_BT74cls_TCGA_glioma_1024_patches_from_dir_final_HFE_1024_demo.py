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

    
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    
    #subfolders = np.array(subfolders)
    return subfolders

def get_case_ids(subfolders):
    
    case_IDS = []
    num_cases = len(subfolders)
    for kk in range(num_cases):
        idv_case_id = subfolders[kk].split('/')[-1]
        case_IDS.append(idv_case_id)
    case_IDS = np.array(case_IDS)
    
    return case_IDS

def matched_case_directory_with_labels(case_ids_final_from_data,survival_logs):
    
    num_cases = len(case_ids_final_from_data)
    ids_from_labels = survival_logs[:,2]
    print('Total training cases : ',num_cases)
    ## For training section.......
    sa_labels_BRAIN = []      
    for k in range(num_cases):
        idv_case_ids = case_ids_final_from_data[k]   
        idx_for_idv_case = np.where(ids_from_labels == idv_case_ids)
        exist_or_not_flag = idv_case_ids in ids_from_labels
        if (exist_or_not_flag == True):   
            sa_labels_BRAIN.append(survival_logs[idx_for_idv_case[0],:])
    
    sa_labels_BRAIN = np.squeeze(np.array(sa_labels_BRAIN))
    return sa_labels_BRAIN
 
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

# BC /home/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/74_clas_DB/test_WSI/BC
parser.add_argument('--input_TCGA_BT_image_path', type=str, default="/home/malom/stjude_projects/computational_pathology/MMBL_CLAS_SA/Histology_analysis/Brain_Kidney_datasets/database/histology_images/BRAIN/TCGA_glioma_1024px_patches/", help='Dataset you are using.')
parser.add_argument('--SA_dataset_path', type=str, default="/research/rgs01/home/clusterHome/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/74_clas_DB/TCAGA_glioma_survival_labels_data/", help='Dataset you are using.')

args = parser.parse_args()

print("creating all the necessary diretoreis:")
# project_log_path_seg = join_path('experimental_logs/', args.project_name_segmentation+'/')   
# training_log_saving_path_seg = join_path(project_log_path_seg,'training/')
# testing_log_saving_path_seg_1 = join_path(project_log_path_seg,'testing/')
# weight_loading_path_seg = join_path(project_log_path_seg,'weights/')

project_log_path_clas = join_path('experimental_logs/', args.project_name_classification+'/')   
training_log_saving_path_clas = join_path(project_log_path_clas,'training/')
testing_log_saving_path_clas = join_path(project_log_path_clas,'testing/')


proj_outputs_name_final = (args.input_TCGA_BT_image_path).split('/')[-2]+'_ouptuts'

if not os.path.isdir("%s/%s"%(testing_log_saving_path_clas,proj_outputs_name_final)):
    os.makedirs("%s/%s"%(testing_log_saving_path_clas,proj_outputs_name_final))        

testing_log_saving_path_final = join_path(testing_log_saving_path_clas,proj_outputs_name_final)


trained_model_path = join_path(project_log_path_clas,'trained_model/')
#weight_loading_path_clas = join_path(project_log_path_clas,'weights/')
# create pathces from the WSI
HPFs_size = (args.HPF_height,args.HPF_width)  
# Step_1 : >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    Extract specific size of the patches       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#pdb.set_trace()
#WSIs_and_patche_dir, directory_names_all = wsi_utils_functions.extract_same_size_patches_from_wsi_with_xy_PIL_SVS_TIF(args.input_svs_image_path, HPFs_size, testing_log_saving_path_final) # Using PIL ......

# Step_2 : >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    Testing the model.       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#pdb.set_trace()
#WSIs_and_patche_dir = testing_log_saving_path_final
#directory_names_all = ['a1980574-357f-11eb-8cd0-001a7dda7111']

###### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Find the case ids from the database path...<<<<<<<<<<<<<<<<<< 
###              This codes need to execute for creating SA and Labels datasets..
path_all_cases = fast_scandir(args.input_TCGA_BT_image_path)
case_ids_final_from_data = get_case_ids(path_all_cases)
# Take case ids, labels, etc from the survival datasets.....
survival_logs = np.load(join_path(args.SA_dataset_path,'final_survival_dataset_processed_for_MML.npy'),allow_pickle=True)
pd_survival_logs = pd.read_csv(join_path(args.SA_dataset_path,'final_survival_dataset_processed_for_MML.csv'))


final_matched_file = matched_case_directory_with_labels(case_ids_final_from_data,survival_logs)
np.save(join_path(args.SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.npy'),final_matched_file)
column_names = pd_survival_logs.columns
column_names_fn = column_names[1:]
pd_final_matched_file = pd.DataFrame(final_matched_file,columns = column_names_fn)
pd_final_matched_file.to_csv(join_path(args.SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.csv'))
###### >>>>>>>>>>>>>>>>LOAD the Final Survival and Labels datasets... <<<<<<<<<<<<<<<<<<<<<<
survival_logs_with_labels415 = np.load(join_path(args.SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.npy'),allow_pickle=True)
pd_survival_logs_with_labels415 = pd.read_csv(join_path(args.SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.csv'))

pdb.set_trace()

WSIs_and_patche_dir = args.input_TCGA_BT_image_path
directory_names_all = pd_survival_logs_with_labels415['patient_ID']

testing_outputs_dir_all, class_logs_directory_name  = wsi_inf.testing_WSIs_from_the_patch_dir(trained_model_path, args.encoded_actual_labels_dir, WSIs_and_patche_dir, directory_names_all, testing_log_saving_path_final)

#pdb.set_trace()

#Step 3 : >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Generate WSI level ouptuts   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< .... 

# num_wsi = len(class_logs_directory_name)-1

# for kk in range(num_wsi):
    
#     WSIs_and_patche_dir_final = os.path.join(WSIs_and_patche_dir,directory_names_all[kk])
#     #wsi_utils_functions.patches_to_image_from_prediction_draw_ROI_WSI_for_BT74class_final( WSIs_and_patche_dir_final, testing_log_saving_path_final)
#     wsi_utils_functions.patches_to_image_from_prediction_draw_ROI_WSI_for_BT74class_final_cv2( WSIs_and_patche_dir_final, testing_log_saving_path_final)
    #pdb.set_trace()
    #wsi_utils_functions.bar_plot_from_the_model_prediction_and_count(WSIs_and_patche_dir_final,directory_names_all[kk],0.9)



# if not os.path.isdir("%s/%s"%(base_dir,'output_images/')):
#     os.makedirs("%s/%s"%(base_dir,'output_images/'))   
# testing_log_saving_path_for_images = join_path(base_dir,'output_images/') 
# wsi_class_obj.patches_to_image_from_prediction_for_BT74class_final(patches_dir,classifier_logs, outputs_saving_dir)



