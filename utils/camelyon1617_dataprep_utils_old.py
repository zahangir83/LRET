# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:53:51 2020

@author: deeplens
"""

#!/usr/bin/env python3
import sys
sys.path.append('/opt/ASAP/bin')

import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
import glob

import numpy as np
import os
import pandas as pd
import json
import random
from PIL import Image
import scipy.io as sio
import cv2
from os import listdir
from os.path import join as join_path
from matplotlib import pyplot as plt

from collections import defaultdict
import csv
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy.ndimage as ndimage
from skimage.transform import resize
import shutil

kernel = np.ones((7,7), np.uint8) 

IMG_CHANNELS = 3
IMG_HEIGHT, IMG_WIDTH = 2084,2084
number_samples_per_images = 200

abspath = os.path.dirname(os.path.abspath(__file__))
allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif','.svs']
current_work_dir=os.getcwd() 
            
import pdb
#please make sure the same number of files in the folder of tumor file and folder of annotation files
#please change the slide_path, anno_path, mask_path accordingly, and leave everything else untouched. 
#pdb.set_trace()

def camelyon_image_annots_to_mask (slide_path,anno_path,camelyon17_type_mask_flag, mask_path):
    #slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/tumor/tumors_051_111/'
    #anno_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/Lesion_annotations/annots/'
    #mask_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/masks/51_110_masks/'  
    tumor_paths = glob.glob(osp.join(slide_path, '*.tif'))
    tumor_paths.sort()
    anno_tumor_paths = glob.glob(osp.join(anno_path, '*.xml'))
    anno_tumor_paths.sort()
    #image_pair = zip(tumor_paths, anno_tumor_paths)  
    #image_pair = list(image_mask_pair)
    #pdb.set_trace()
    reader = mir.MultiResolutionImageReader()
    i=0
    while i < len(tumor_paths):
        mr_image = reader.open(tumor_paths[i])
        annotation_list=mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        xml_repository.setSource(anno_tumor_paths[i])
        xml_repository.load()
        annotation_mask=mir.AnnotationToMask()
        camelyon17_type_mask = camelyon17_type_mask_flag
        label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 255, '_1': 255, '_2': 0}
        conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
        output_path= osp.join(mask_path, osp.basename(tumor_paths[i]).replace('.tif', '_mask.tif'))
        annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
        i=i+1


def extract_image_seq_non_overlapped_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):   
        
    height,width = full_mask.shape     
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]                
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.jpg'           
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)            
            mx_val = patch_mask.max()
            mn_val = patch_mask.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))            
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
#            if mx_val > 0:
#                cv2.imwrite(final_des_img,patch_img)
#                cv2.imwrite(final_des_mask,patch_mask)
            pn+=1            
        k +=1
        print ('Processing for: ' +str(k))        
    return pn

def data_aug_and_save_for_classification(full_imgs,img_name,num_agu_per_sample, img_saving_dir):
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = full_imgs  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=img_saving_dir, save_prefix=img_name, save_format='png'):
        i += 1
        if i > num_agu_per_sample:
            break  # otherwise the generator would loop indefinitely
            
def extract_image_random_patches(full_imgs,mask_img,central_xy, N_patches, patch_h,patch_w, img_name, imd_saving_dir):
       
    central_xy.astype(int)    
    height, width, chan = full_imgs.shape   
    start_x = patch_h/2
    start_y = patch_w/2
    end_x = height-start_x
    end_y = width - start_y    
    k=0
    pn = 0
    while k <N_patches:
        x_center = int(central_xy[k,0])
        y_center =  int(central_xy[k,1])        
        if (x_center > start_x and y_center > start_y and x_center < end_x and y_center < end_y) :
            img_patch = full_imgs[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2),:]
            mask_patch = mask_img[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask.jpg'
            final_img_des = os.path.join(imd_saving_dir,f_img_name)
            final_mask_des = os.path.join(imd_saving_dir,f_mask_name)
            mx_val = final_mask_des.max()
            mn_val = final_mask_des.min()           
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))  
            cv2.imwrite(final_img_des,img_patch)
            cv2.imwrite(final_mask_des,mask_patch)
            pn +=1   
        k +=1  
    print ('Processing for: ' +str(k))    
    return pn
               
def create_training_data_from_sub_sub_dir(data_path, patch_h, patch_w, acc_name, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):        
        for name in subdirs:            
            sub_dir = os.path.join(path, name+'/')            
            print(name)             
            for path_2, subdirs_2,files_2 in os.walk(sub_dir):                
                for name_2 in subdirs_2:                    
                    sub_sub_dir = os.path.join(path_2, name_2+'/')                    
                    images = os.listdir(sub_sub_dir)                    
                    if name_2 =='0':
                        for image_name in images:  
                            read_img = cv2.imread(os.path.join(sub_sub_dir,image_name))
                            h,w,c=read_img.shape
                            if h==w:
                                shutil.copy(os.path.join(sub_sub_dir,image_name),os.path.join(img_saving_dir+'0/'))
                    else:
                        for image_name in images:                           
                            read_img = cv2.imread(os.path.join(sub_sub_dir,image_name))
                            h,w,c=read_img.shape
                            if h==w:
                                shutil.copy(os.path.join(sub_sub_dir,image_name),os.path.join(img_saving_dir+'1/'))
                    
                   
def create_training_images_masks_from_sub_dir(data_path, patch_h, patch_w, img_saving_dir):
    
    for path, subdirs, files in os.walk(data_path):       
        for name in subdirs:           
            sub_dir = os.path.join(path, name+'/')           
            print(name)            
            images = [x for x in sorted(os.listdir(sub_dir)) if x[-9:] == '_mask.jpg']   
            total = np.round(len(images))                
            i = 0
            #pdb.set_trace()
            print('Creating training images...')               
            for image_mask_name in images:
                image_name = image_mask_name.split('_mask')[0] + '.jpg'       
                acc_name = image_name.split('.')[0]
                image = cv2.imread(os.path.join(sub_dir, image_name),cv2.IMREAD_UNCHANGED)
                mask_im = cv2.imread(os.path.join(sub_dir, image_mask_name), cv2.IMREAD_GRAYSCALE)
                num_patches = extract_image_seq_non_overlapped_patches (image, mask_im, patch_h, patch_w, acc_name, img_saving_dir)
                
                print ('Processing done for: ' +str(i))

def create_patches_masks_from_dir(IMAGE_PATH,patch_h, patch_w, img_saving_dir):         
    train_data_path = os.path.join(IMAGE_PATH)
    images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))   
    total = np.round(len(images))    
    i = 0
    print('Creating training images...')   
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'       
        acc_name = image_name.split('.')[0]
        im = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_UNCHANGED)
        mask_im = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_rz = im
        img_mask_rz = mask_im 
        num_patches = extract_image_seq_non_overlapped_patches (img_rz, img_mask_rz, patch_h, patch_w, acc_name, img_saving_dir)       
        print ('Processing for: ' +str(i))    
    return 0

def create_dataset_random_patches_driver(base_dir):    
    train_data_path = os.path.join(base_dir)
    images = filter((lambda image: '_anno' not in image), os.listdir(train_data_path))
    total = np.round(len(images)) 
    for filename in images:
        img = cv2.imread(os.path.join(base_dir,filename))            
        img_name = filename.split('/')[-1]
        img_name = img_name.split('.')[0]
        mask_name = img_name+'_anno.bmp'
        mask_path = os.path.join(base_dir,mask_name)           
        mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)                   
        height, width, chan = img.shape           
        radius = min((height, width))           
        central_xy = np.random.random((number_samples_per_images, 2))*radius             
        central_xy=central_xy.astype(int)
        print(img_name)                        
        img_saving_dir_b = os.path.join(data_dir,'images_and_masks_benign_malignant/benign/')  
        if len(central_xy) > 0:
               num_patches = extract_image_random_patches (img, mask_img, central_xy, len(central_xy), patch_h, patch_w, img_name, img_saving_dir_b)
    return 0


def refine_patches_masks_create_final_samples(image_dir, TP_images_dir, patch_h, patch_w, img_saving_dir):   

    TP_images_path_final = glob.glob(osp.join(TP_images_dir, '*.png'))
    TP_images_path_final.sort()
    num_tp_samples = np.round(len(TP_images_path_final))
      
    train_data_path = os.path.join(image_dir)
    images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))   
    total = np.round(len(images))    
    i = 0
    print('Creating training images...')   
    for image_name in images:
        if 'mask' in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.png'       
        #acc_name = image_name.split('.')[0]
        patch_img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_UNCHANGED)
        mask_h,mask_w,c=patch_img.shape
        patch_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        mx_val = patch_mask.max()
        print ('max_val : '+str(mx_val))
 
        final_des_img = os.path.join(img_saving_dir,image_name)
        final_des_mask = os.path.join(img_saving_dir,image_mask_name)   
                       
        #pdb.set_trace()
        
        if mx_val > 0:
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
        else:
            image_id_tp = random.randint(0,num_tp_samples-1)    
            tp_img_path = TP_images_path_final[image_id_tp] 
            tp_im = cv2.imread(os.path.join(tp_img_path),cv2.IMREAD_UNCHANGED)
            
            random_size = random.randint(32,78)
            start_x = random.randint(1,mask_h-random_size)   
            start_y = random.randint(1,mask_w-random_size)
            
            patch_img[start_y: start_y+random_size,start_x: start_x+random_size,:] = tp_im[start_y: start_y+random_size,start_x: start_x+random_size,:]
            child_mask = 255.0* np.ones((random_size,random_size), dtype='float32')
            patch_mask[start_y: start_y+random_size,start_x: start_x+random_size] = child_mask
            
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
   
        print ('Processing for: ' +str(i))    
        
    return 0



def create_seg_mask_for_normal_images(normal_tissue_dir, TP_images_dir, image_mask_saving_dir):   
    
    TP_images_path_final = glob.glob(osp.join(TP_images_dir, '*.jpg'))
    TP_images_path_final.sort()
    num_tp_samples = np.round(len(TP_images_path_final))
    
    images = [x for x in sorted(os.listdir(normal_tissue_dir)) if x[-4:] == '.png']
    total_samples = np.round(len(images))    
    i = 0
    print('Creating refined images and masks....')
    for image_name in images:
        
        #pdb.set_trace()
        
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'       
        #acc_name = image_name.split('.')[0]

        acc_im = cv2.imread(os.path.join(normal_tissue_dir, image_name),cv2.IMREAD_UNCHANGED)
        mask_h,mask_w,c=acc_im.shape
        mask_initial = np.zeros((mask_h,mask_w), dtype='float32')
        
        image_id_tp = random.randint(0,num_tp_samples-1)    
        tp_img_path = TP_images_path_final[image_id_tp] 
        tp_im = cv2.imread(os.path.join(tp_img_path),cv2.IMREAD_UNCHANGED)
        
        random_size = random.randint(32,78)
    
        start_x = random.randint(1,mask_h-random_size)   
        start_y = random.randint(1,mask_w-random_size)
        
        acc_im[start_y: start_y+random_size,start_x: start_x+random_size,:] = tp_im[start_y: start_y+random_size,start_x: start_x+random_size,:]
        child_mask = 255.0* np.ones((random_size,random_size), dtype='float32')
        mask_initial[start_y: start_y+random_size,start_x: start_x+random_size] = child_mask
          
        mx_val = mask_initial.max()
        mn_val = mask_initial.min()            
        print ('max_val : '+str(mx_val))
        print ('min_val : '+str(mn_val)) 

        final_des_img = os.path.join(image_mask_saving_dir,image_name)
        final_des_mask = os.path.join(image_mask_saving_dir,image_mask_name)                
      
        if mx_val > 0:
            cv2.imwrite(final_des_img,acc_im)
            cv2.imwrite(final_des_mask,mask_initial)



def extract_HPFs_mask_for_tumor(full_img,full_mask,patch_h,patch_w, img_name, img_saving_dir):
    
    if not os.path.isdir("%s/%s"%(img_saving_dir,img_name)):
        os.makedirs("%s/%s"%(img_saving_dir,img_name))        
    
    patches_saving_dir = join_path(img_saving_dir+img_name+'/')
    
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)

    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.jpg'           
            final_des_img = os.path.join(patches_saving_dir,f_img_name)
            final_des_mask = os.path.join(patches_saving_dir,f_mask_name)
            
            mx_val = patch_mask.max()
            mn_val = patch_mask.min()
            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))     

            if mx_val > 10:
                cv2.imwrite(final_des_img,patch_img)
                cv2.imwrite(final_des_mask,patch_mask)
            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))

    return pn


def extract_HPFs_from_normal(full_img, patch_h, patch_w, img_name, img_saving_dir):
    
    if not os.path.isdir("%s/%s"%(img_saving_dir,img_name)):
        os.makedirs("%s/%s"%(img_saving_dir,img_name))        
    
    patches_saving_dir = join_path(img_saving_dir+img_name+'/')
    
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)    
    #pdb.set_trace()
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            final_des_img = os.path.join(patches_saving_dir,f_img_name)
            cv2.imwrite(final_des_img,patch_img)
            pn+=1     
        k +=1
        print ('Processing for: ' +str(k))

    return pn

def extract_same_size_patches_from_svs(svs_img_dir, patches_saving_dir, patch_size):
    
    '''        
    patch_dir_name = 'patches_'+str(patch_size[0])+'/ 
    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    '''
    patches_dir = patches_saving_dir
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.tif']
        
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print(svs_img_dir.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir[0]+dir_name+'/')
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
                
                idx_sr_sc = str(img_saving_idx)+','+str(x)+','+str(y)                
                starting_row_columns.append(idx_sr_sc)
                print("Processing:"+str(img_saving_idx))                
                ac_img_name =str(img_saving_idx)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                #cv2.imwrite(final_img_des,img)
                imsave(final_img_des,img)                
                img_saving_idx +=1
                
        scan.close    
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

def extact_HPFs_patches_mask_from_tumor(wsi_path, wsi_mask_path,patch_h,patch_w, HPFS_saving_dir):
    
    #slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/tumor/wsi'
    #mask_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/masks/new_masks'
    slide_paths = glob.glob(osp.join(wsi_path, '*.tif'))
    slide_paths.sort()
    mask_paths = glob.glob(osp.join(wsi_mask_path, '*.tif'))
    mask_paths.sort()
    
    #HPFS_saving_dir = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/HPFs_images_masks_camelyon16/'
    
    patch_h =1024
    patch_w = 1024
    #full_img = rgb_imagenew
    #full_mask = grey
    
    scan_id=0
    while scan_id < len(slide_paths):
        f = slide_paths[scan_id]
        f_mask = mask_paths[scan_id]
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue       
        
        img_name_w_ext = f.split('/')[-1]
        img_name = img_name_w_ext.split('.')[0]
        
        slide = openslide.open_slide(f)
        truth = openslide.open_slide(f_mask)
    
        lavel = 2
        rgb_image = slide.read_region((0, 0), lavel, slide.level_dimensions[2])
        rgb_mask = truth.read_region((0, 0), lavel, slide.level_dimensions[2])
    
        rgb_imagenew = np.array(rgb_image)
        mask_image = np.array(rgb_mask.convert('L'))
       
        print('Processing for image id: ' + str(img_name))
        extract_HPFs_mask_for_tumor(rgb_imagenew,mask_image,patch_h,patch_w, img_name, HPFS_saving_dir)
        scan_id = scan_id + 1


def extact_HPFS_patches_from_normal(wsi_slide_path, patch_h, patch_w, HPFs_saving_dir):
    
    #slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/normal'
    slide_paths = glob.glob(osp.join(wsi_slide_path, '*.tif'))
    slide_paths.sort()
    #img_saving_dir = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/HPFs_images_masks_camelyon16/normal/'
    #patch_h =1024
    #patch_w = 1024
    scan_id=0
    while scan_id < len(slide_paths):
        f = slide_paths[scan_id]
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue              
        img_name_w_ext = f.split('/')[-1]
        img_name = img_name_w_ext.split('.')[0]        
        slide = openslide.open_slide(f)    
        lavel = 2
        rgb_image = slide.read_region((0, 0), lavel, slide.level_dimensions[2])
        rgb_imagenew = np.array(rgb_image)
        print('Processing for image id: ' + str(img_name))
        extract_HPFs_from_normal(rgb_imagenew,patch_h,patch_w, img_name, HPFs_saving_dir)
        scan_id = scan_id + 1



def main():
    
    # read the whole slide image and create the mask...
    slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/tumor/tumors_051_111/'
    anno_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/Lesion_annotations/annots/'
    mask_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/masks/51_110_masks/' 
    
    camelyon17_type_mask_flag = False
    camelyon_image_annots_to_mask (slide_path,anno_path,camelyon17_type_mask_flag, mask_path):

    # extract the HPFs from Wholse Slide Images....

    # For tumor slide to HPFs....
    wsi_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/tumor/tumors_051_111/'
    wsi_mask_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/masks/51_110_masks/' 
    HPFs_saving_dir =  ' ... path for saving HPFS... for tumor...'
    HPFs_patch_h = 2048
    HPFs_patch_w = 2048
    extact_HPFs_patches_mask_from_tumor(wsi_path, wsi_mask_path,patch_h,patch_w, HPFs_saving_dir):
    # For normal Whole slide images...
        
    wsi_slide_path = 'Path for normal wsi'
    HPFs_saving_dir = ' HPFs saving dir  for normal..'
    HPFs_patch_h = 2048
    HPFs_patch_w = 2048
    extact_HPFS_patches_from_normal(wsi_slide_path, HPFs_patch_h, HPFs_patch_w, HPFs_saving_dir) 
    
    
    # Extract the sub-patches from HPFs...

    data_path = join_path(abspath, 'HPFs_images_masks_camelyon16/tumor/training/')
    patches_saving_dir = join_path(abspath, 'HPFs_images_masks_camelyon16/tumor/training_patches/')
    patch_h = 128
    patch_w = 128
    #create_patches_masks_from_dir(data_path, patch_h, patch_w, patches_saving_dir)
    create_training_images_masks_from_sub_dir(data_path, patch_h, patch_w, patches_saving_dir)
      
      
    # Creating masks for normal referenes..only..
    normal_tissue_dir = join_path(abspath, 'example_images/normal/')
    TP_images_dir = join_path(abspath, 'example_images/TP/')
    image_mask_saving_dir = join_path(abspath, 'example_images/create_images_masks/')
    #create_seg_mask_for_normal_images(normal_tissue_dir, TP_images_dir, image_mask_saving_dir)
      
      
    # Refined images and masks .......
    image_mask_dir = join_path(abspath, 'example_images/patches_from_HPFs/')
    TP_images_dir = join_path(abspath, 'example_images/TP/')
    refined_image_mask_saving_dir = join_path(abspath,'example_images/refine_patches_mask_from_HPFs/')
    #refine_patches_masks_create_final_samples(image_mask_dir, TP_images_dir, patch_h, patch_w, refined_image_mask_saving_dir)
      
      
    
    

if __name__== "__main__": 
    # call the main function..
    main()
