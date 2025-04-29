#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:44:49 2020

@author: malom
"""
import numpy as np
import dataset_utils as data_utils
import data_management as data_mgt
import glob
import shutil
import os
import pdb


if __name__ == '__main__':
    
    ## >>>>>>>>>>>>>    Colon cancer dataset preparation <<<<<<<<<<<<<<<<<<<<<<
    
    # To create sub-patches from the directories....
    # image_path = '/home/malom/stjude_projects/digital_path/colon_cancer_detection/database/train_val/'
    # patches_saving_path = '/home/malom/stjude_projects/digital_path/colon_cancer_detection/database/train_val_patches_128/'
    # patch_h =128 
    # patch_w = 128
    # number_samples_per_images = 10 
    # #pdb.set_trace()
    #data_utils.create_dataset_random_patches_driver(image_path,patch_h,patch_w,number_samples_per_images, patches_saving_path)
    
    ## >>>>>>>>>>>>>    74-class problem dataset preparation <<<<<<<<<<<<<<<<<<<<<<
    # Copying the files for 74 classes.......from data center to HPC data storgae .....
    #src_dir = '/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/74_classes_database/Final_dataset/'
    #dst_dir = '/scratch_space/malom/project_74_classes/db_final/'
    
    #pdb.set_trace()
    num_samples = 5000
    #data_mgt.copy_specific_num_images_from_sub_dir_train_val_test(src_dir,num_samples, dst_dir)
    #data_mgt.copy_images_from_sub_dir(src_dir,dst_dir)
    

    # For creating the subset of training and testing samples for final testing.....
    
    
    scr_dir = '/scratch_space/malom/project_74_classes/db_final/test/'
    dst_dir = '/scratch_space/malom/project_74_classes/db_final/test_subset/'
    
    num_samples = 50
    #data_mgt.copy_specific_num_images_to_sub_directories(scr_dir,num_samples,dst_dir)

    #scr_dir = '/scratch_space/malom/project_74_classes/db_final/val/'
    #dtl_ref_dir = '/scratch_space/malom/project_74_classes/db_final/test/'
    #data_mgt.delete_redundant_image_samples(scr_dir,dtl_ref_dir)
    
    # Selecting the files....
    scr_dir = '/scratch_space/malom/project_74_classes/db_final/test/'
    numpy_file_dir = '/home/malom/stjude_projects/digital_path/74_classes_project/project_74_classes_digital_path/experimental_logs/project_74cls_ResNet_BBBL/results/test_all_samples/'
    dst_dir = '/home/malom/stjude_projects/digital_path/74_classes_project/project_74_classes_digital_path/experimental_logs/project_74cls_ResNet_BBBL/results/test_final/'
    #data_mgt.copying_dl_fv_conf_vec_for_samples(scr_dir,numpy_file_dir, dst_dir)
    
    
    
    scr_dir = '/research/rgs01/home/clusterHome/malom/stjude_projects/digital_path/74_classes_project/project_74_classes_digital_path/experimental_logs/project_74cls_ResNet_BBBL/results/0/'
    dtl_ref_dir = '/scratch_space/malom/project_74_classes/db_final/train/0/'
    #data_mgt.delete_redundant_samples_from_dl_fv_class_calls(scr_dir,dtl_ref_dir)
    
    # delete random number of samples..
    #dtl_ref_dir = '/scratch_space/malom/project_74_classes/db_final/train/0/'
    #num_samples = 533
    #data_mgt.delete_random_number_samples(dtl_ref_dir, num_samples)
    
    # Count and save number of samples used for training
    scr_dir = '/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/74_classes_database/Final_dataset//'
    #scr_dir = '/scratch_space/malom/project_74_classes/db_final/test/'
    data_logs_saving_dir = '/research/rgs01/home/clusterHome/malom/stjude_projects/digital_path/74_classes_project/project_74_classes_digital_path/experimental_logs/project_74cls_ResNet_BBBL/results/samples_logs/'
    file_name = 'actual_database'
    #data_mgt.record_num_sample_per_class(scr_dir,data_logs_saving_dir,file_name)


    ## >>>>>>>>>>>>>    dataset preparation for CycleGAN <<<<<<<<<<<<<<<<<<<<<<

    src_dir = '/research/rgs01/project_space/orrgrp/medulloblastoma/common/MZA/cycleGAN_dataset/'
    dst_dir = '/home/malom/stjude_projects/digital_path/74_classes_project/stain_normalization/cycleGAN_DB/'   
    data_mgt.copy_images_from_sub_dir_with_diff_ext(src_dir,dst_dir)

    