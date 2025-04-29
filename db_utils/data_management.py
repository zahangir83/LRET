#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:57:18 2020

@author: malom
"""

import numpy as np
import dataset_utils as data_utils
from os.path import join as join_path
import glob
import cv2
import shutil
import random

import os,json
import pdb
from PIL import Image

from skimage import io

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
def find_files_numbers(data_dir):
    
    files_number = []
    dirctory_names = []
    for path, subdirs, files in os.walk(data_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(data_dir, dir_name+'/')           
            print(dir_name) 
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            number_files = len(image_list)
            files_number.append(number_files)
            dirctory_names.append(dir_name)
    
    np_files_number = np.array(files_number)
    # find min number of files
    lowest_files_num = np.amin(np_files_number) 

    return np_files_number, lowest_files_num
    
                
def copy_images_from_dir(src_dir,dst_dir):
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        shutil.copy(jpgfile, dst_dir)
        

def copy_images_from_sub_dir(scr_dir,dst_dir):
    
    #pdb.set_trace()
    
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(dst_dir,dir_name)):
                os.makedirs("%s/%s"%(dst_dir,dir_name))                
            dst_dir_final = join_path(dst_dir,dir_name+'/')
            
            for jpgfile in glob.iglob(os.path.join(sub_dir_path, "*.jpg")):
                print('Copying files for : ',dir_name )
                shutil.copy(jpgfile, dst_dir_final)

def copy_images_from_sub_dir_with_diff_ext(scr_dir,dst_dir):
    
    #pdb.set_trace()
    
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(dst_dir,dir_name)):
                os.makedirs("%s/%s"%(dst_dir,dir_name))                
            dst_dir_final = join_path(dst_dir,dir_name+'/')            
            #jpgfile = glob.iglob(os.path.join(sub_dir_path, "*.jpg" or "*.png"))
            image_list = os.listdir(sub_dir_path) # dir is your directory path

            for i, img_name in enumerate(image_list): 
                jpgfile = join_path(sub_dir_path,img_name)                
                img_ext = img_name.split('.')[1]
                if img_ext =='jpg' or 'png':
                    check_valid_image_flag = verify_image(jpgfile)
                    if check_valid_image_flag == True:
                        img = cv2.imread(jpgfile, cv2.IMREAD_COLOR)   
                        #imgp = Image.open(jpgfile) # open the image file
                        #imgp.verify() # verify that it is, in fact an image
                        img_name_owext = img_name.split('.')[0]
                        img_name_4s = img_name_owext+'.png'
                        print('Copying files for : ',img_name_owext )                                 
                        final_dst = join_path(dst_dir_final, img_name_4s)
                        cv2.imwrite(final_dst,img)    

                
def copy_specific_num_images_from_sub_dir_train_test(scr_dir,num_samples, dst_dir_train, dst_dir_test):
    
     
    #np_files_number, lowest_files_num = find_files_numbers(scr_dir)
    #num_samples = lowest_files_num
    
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(dst_dir_train,dir_name)):
                os.makedirs("%s/%s"%(dst_dir_train,dir_name))                
            dst_dir_train_final = join_path(dst_dir_train,dir_name+'/')
            
            if not os.path.isdir("%s/%s"%(dst_dir_test,dir_name)):
                os.makedirs("%s/%s"%(dst_dir_test,dir_name))                
            dst_dir_test_final = join_path(dst_dir_test,dir_name+'/')
            #pdb.set_trace() 
            #image_files = glob.iglob(os.path.join(sub_dir_path, "*.jpg"))
            #images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.jpg'] 
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            total_samples = len(image_list)
            #perm = np.random.permutation(total_samples)
            #image_files_shuffled = image_list[perm]
            image_files_np = np.array(image_list)
            print('Totla number of samples:', total_samples)
            if total_samples > num_samples:
                images_files_final = image_files_np[:num_samples]  
            else:
                images_files_final = image_files_np
            
            number_samples_total = len(images_files_final)           
            train_samples = int(number_samples_total*0.8)           
            #val_test_samples = number_samples_total-train_samples
            
            
            #for jpgfile in images_files_final:
            for i, img_name in enumerate(images_files_final): 
                print('Copying files for : ',dir_name )
                jpgfile = join_path(sub_dir_path,img_name)   
                
                if i < train_samples:
                    shutil.copy(jpgfile, dst_dir_train_final)
                else:
                    shutil.copy(jpgfile, dst_dir_test_final)

def copy_specific_num_images_from_sub_dir_train_val_test(scr_dir,num_samples,dst_dir):
    

    if not os.path.isdir("%s/%s"%(dst_dir,'train')):
                os.makedirs("%s/%s"%(dst_dir,'train'))                
    dst_dir_train = join_path(dst_dir,'train'+'/')
    
    if not os.path.isdir("%s/%s"%(dst_dir,'val')):
                os.makedirs("%s/%s"%(dst_dir,'val'))                
    dst_dir_val = join_path(dst_dir,'val'+'/')  

    if not os.path.isdir("%s/%s"%(dst_dir,'test')):
                os.makedirs("%s/%s"%(dst_dir,'test'))                
    dst_dir_test = join_path(dst_dir,'test'+'/')        
    #np_files_number, lowest_files_num = find_files_numbers(scr_dir)
    #num_samples = lowest_files_num
    
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print(dir_name)    
            
            if not os.path.isdir("%s/%s"%(dst_dir_train,dir_name)):
                os.makedirs("%s/%s"%(dst_dir_train,dir_name))                
            dst_dir_train_final = join_path(dst_dir_train,dir_name+'/')
            
            if not os.path.isdir("%s/%s"%(dst_dir_val,dir_name)):
                os.makedirs("%s/%s"%(dst_dir_val,dir_name))                
            dst_dir_val_final = join_path(dst_dir_val,dir_name+'/')
            
            if not os.path.isdir("%s/%s"%(dst_dir_test,dir_name)):
                os.makedirs("%s/%s"%(dst_dir_test,dir_name))                
            dst_dir_test_final = join_path(dst_dir_test,dir_name+'/')
            #pdb.set_trace() 
            #image_files = glob.iglob(os.path.join(sub_dir_path, "*.jpg"))
            #images = [x for x in sorted(os.listdir(sub_dir)) if x[-4:] == '.jpg'] 
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            total_samples = len(image_list)
            #perm = np.random.permutation(total_samples)
            #image_files_shuffled = image_list[perm]
            image_files_np = np.array(image_list)
            print('Totla number of samples:', total_samples)       
            #pdb.set_trace()
            
            if total_samples > num_samples:
                #images_files_final = image_files_np[:num_samples]  
                rand_sample_idxs = random.sample(range(0,total_samples),num_samples)
                images_files_final = image_files_np[rand_sample_idxs]  

            else:
                images_files_final = image_files_np
            
            number_samples_total = len(images_files_final)           
            train_samples = int(number_samples_total*0.8)           
            val_test_samples = number_samples_total-train_samples
            print(val_test_samples)
            val_last_sample = train_samples+int(val_test_samples*0.5)
        
            #for jpgfile in images_files_final:
            for i, img_name in enumerate(images_files_final): 
                #print('Copying files for : ',dir_name )
                jpgfile = join_path(sub_dir_path,img_name)   
                print(jpgfile)
                img_ext = img_name.split('.')[1]
                
                #pdb.set_trace()
                if img_ext =='jpg':
                    
                    check_valid_image_flag = verify_image(jpgfile)
                    
                    if check_valid_image_flag == True:
                            
                        img = cv2.imread(jpgfile, cv2.IMREAD_COLOR)   
                        
                        #imgp = Image.open(jpgfile) # open the image file
                        #imgp.verify() # verify that it is, in fact an image
          
                        img_name_owext = img_name.split('.')[0]
                        img_name_4s = img_name_owext+'.png'
                        print('Copying files for : ',img_name_owext )
                  
                            
                        if i < train_samples:
                            #shutil.copy(jpgfile, dst_dir_train_final)
                            final_dst_train = join_path(dst_dir_train_final, img_name_4s)
                            cv2.imwrite(final_dst_train,img)
                            #imgp.save(final_dst_train)
                        elif(i < val_last_sample):
                            #shutil.copy(jpgfile, dst_dir_val_final)
                            final_dst_val = join_path(dst_dir_val_final, img_name_4s)
                            cv2.imwrite(final_dst_val,img)
                            #imgp.save(final_dst_val)
    
                        else:
                            #shutil.copy(jpgfile, dst_dir_test_final)
                            final_dst_test = join_path(dst_dir_test_final, img_name_4s)
                            cv2.imwrite(final_dst_test,img)
                            #imgp.save(final_dst_test)
                            

def copy_specific_num_images_to_sub_directories(scr_dir,num_samples,dst_dir):
       
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print(dir_name)                
            if not os.path.isdir("%s/%s"%(dst_dir,dir_name)):
                os.makedirs("%s/%s"%(dst_dir,dir_name))                
            dst_dir_final = join_path(dst_dir,dir_name+'/')            
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            total_samples = len(image_list)
            image_files_np = np.array(image_list)
            print('Totla number of samples:', total_samples)                   
            if total_samples > num_samples:
                #images_files_final = image_files_np[:num_samples]  
                rand_sample_idxs = random.sample(range(0,total_samples),num_samples)
                images_files_final = image_files_np[rand_sample_idxs]  
            else:
                images_files_final = image_files_np   
                                
            for i, img_name in enumerate(images_files_final): 
                #print('Copying files for : ',dir_name )
                jpgfile = join_path(sub_dir_path,img_name)   
                print(jpgfile)
                img_ext = img_name.split('.')[1]                
                #pdb.set_trace()
                if img_ext =='png':                  
                    check_valid_image_flag = verify_image(jpgfile)                    
                    if check_valid_image_flag == True:                            
                        img = cv2.imread(jpgfile, cv2.IMREAD_COLOR)   
                        img_name_owext = img_name.split('.')[0]
                        img_name_4s = img_name_owext+'.png'
                        print('Copying files for : ',img_name_owext )                  
                        final_dst = join_path(dst_dir_final, img_name_4s)
                        cv2.imwrite(final_dst,img)


def record_num_sample_per_class(scr_dir,data_logs_saving_dir,file_name):

    
    #pdb.set_trace()
    class_names = []
    num_samples_per_class = []   
    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
            print('Class name: ', dir_name)    
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            total_samples = len(image_list)
            print('Totla number of samples:', total_samples)              
            class_names.append(dir_name)
            num_samples_per_class.append(total_samples)
            
    # saving the log values ....
    class_names = np.array(class_names)
    num_samples_per_class = np.array(num_samples_per_class)
    
    
    samples_log = {}
    samples_log["Class_names"] = class_names
    samples_log["Num_samples"] = num_samples_per_class
    json_file = os.path.join(data_logs_saving_dir,file_name+'_sample_logs.json')
    with open(json_file, 'w') as file_path:
        json.dump(samples_log, file_path, indent=4, sort_keys=True,cls=NumpyEncoder)
        
        
def delete_redundant_image_samples(scr_dir,dtl_ref_dir):

    #dir_names = []

    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')   
            dlt_dir_path = os.path.join(dtl_ref_dir, dir_name+'/')           
            #print('Class name: ', dir_name)    
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            dlt_image_list = os.listdir(dlt_dir_path) # dir is your directory path
            
            #pdb.set_trace()
            flag = False
            count = 0
            for k in range(len(dlt_image_list)):          
                img_name = dlt_image_list[k]         
                if img_name in image_list:
                    os.remove(dlt_dir_path + '/' + img_name)
                    #print('Deleting the file name :', img_name)
                    flag = True
                    count = count+1
            
            if flag == True:
                print('Redundant file founds :', dir_name)
                print('Number of samples: ', count)
                #dir_names.append(dir_name)

def delete_random_number_samples(dtl_ref_dir, num_samples):
    
    for path, subdirs, files in os.walk(dtl_ref_dir):        
        for dir_name in subdirs:            
            dlt_dir_path = os.path.join(dtl_ref_dir, dir_name+'/')           
            dlt_image_list = os.listdir(dlt_dir_path) # dir is your directory path
            dlt_image_list = np.array(dlt_image_list)
            #pdb.set_trace()
            total_samples = int(len(dlt_image_list))
            rand_sample_idxs = random.sample(range(0,total_samples),num_samples)
            images_files_final = dlt_image_list[rand_sample_idxs]  

            for k in range(len(images_files_final)):          
                img_name = images_files_final[k]         
                os.remove(dlt_dir_path + '/' + img_name)
                  
                 
def delete_redundant_samples_from_dl_fv_class_calls(scr_dir,dtl_ref_dir):

    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')   
            dlt_dir_path = os.path.join(dtl_ref_dir, dir_name+'/')           
            print('Class name: ', dir_name)                
            file_names = [x for x in sorted(os.listdir(sub_dir_path)) if x[-14:] == '_fm_vector.npy'] 
            dlt_image_list = os.listdir(dlt_dir_path) # dir is your directory pat      
            pdb.set_trace()
            
            counter  = 0
            for k in range(len(file_names)):          
                file_name = file_names[k]  
                img_name_woext = file_name.split('_fm_vector')[0]
                img_name = img_name_woext+'.png'
                if img_name in dlt_image_list:
                    os.remove(dlt_dir_path + '/' + img_name)
                    print('Found the redundant file name :', img_name)
                    counter = counter+1
            print('Total number of samples are deleted : ', counter)

                    
def copying_dl_fv_conf_vec_for_samples(scr_dir,numpy_file_dir, dst_dir):

    for path, subdirs, files in os.walk(scr_dir):        
        for dir_name in subdirs:            
            sub_dir_path = os.path.join(scr_dir, dir_name+'/')   
            numpy_file_dir_final = os.path.join(numpy_file_dir, dir_name+'/')  
            
            if not os.path.isdir("%s/%s"%(dst_dir,dir_name)):
                os.makedirs("%s/%s"%(dst_dir,dir_name))                
            dst_dir_final = join_path(dst_dir,dir_name+'/')
            
            print('Class name: ', dir_name)    
            image_list = os.listdir(sub_dir_path) # dir is your directory path
            
            #pdb.set_trace()
            for k in range(len(image_list)):          
                img_name = image_list[k]  
                img_name_woext = img_name.split('.')[0]
                fv_file_name = img_name_woext+'_fm_vector.npy'
                conf_file_name = img_name_woext+'_class_prob.npy'
                #if img_name in image_list:
                fv_npy_file = numpy_file_dir_final + '/' + fv_file_name
                shutil.copy(fv_npy_file, dst_dir_final)
                cv_npy_file = numpy_file_dir_final + '/' + conf_file_name
                shutil.copy(cv_npy_file, dst_dir_final)
                print('Copying the file name :', img_name)
               