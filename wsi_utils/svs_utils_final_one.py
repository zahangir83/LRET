# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:54:37 2019
@author: deeplens
"""
import openslide
#from scipy.misc import imsave, imresize
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np
import os
import pdb
import json
import cv2
from scipy.ndimage import binary_fill_holes
from skimage import filters, img_as_ubyte
from skimage.morphology import remove_small_objects
#from scipy.misc import imsave
from os.path import join as join_path
from PIL import Image
import pyvips
import math
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

import zipfile
import zlib # important for best compression size

def read_and_preprocess_img(path, size=(224,224)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return x



abspath = os.path.dirname(os.path.abspath(__file__))
#kernel = np.ones((5,5), np.uint8) 
kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) ## create ellipse like kernel
#wsi_kernel = np.ones((55,55), np.uint8) 
wsi_kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))  ## create ellipse like kernel

ROI_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27))  ## create ellipse like kernel
save = True

#valid_images = ['.svs','.jpg','.tif']
valid_images = ['.tif','.jpg','.svs']

def create_heatmap(im_map, im_cloud, kernel_size=(3,3),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    ''' 
    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)
    return (a1*im_map + a2*im_cloud_clr).astype(np.uint8) 

def rgb2gray(img):
    """Convert RGB image to gray space.
    Parameters
    ----------
    img : np.array
        RGB image with 3 channels.
    Returns
    -------
    gray: np.array
        Gray image
    """
    gray = np.dot(img, [0.299, 0.587, 0.114])

    return gray


def thresh_slide(gray, thresh_val, sigma=13):
    """ Threshold gray image to binary image
    Parameters
    ----------
    gray : np.array
        2D gray image.
    thresh_val: float
        Thresholding value.
    smooth_sigma: int
        Gaussian smoothing sigma.
    Returns
    -------
    bw_img: np.array
        Binary image
    """

    # Smooth
    smooth = filters.gaussian(gray, sigma=sigma)
    smooth /= np.amax(smooth)
    # Threshold
    bw_img = smooth < thresh_val

    return bw_img


def fill_tissue_holes(bw_img):
    """ Filling holes in tissue image
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    Returns
    -------
    bw_fill: np.array
        Binary image with no holes
    """

    # Fill holes
    bw_fill = binary_fill_holes(bw_img)

    return bw_fill

def perform_morphological_operations(pred):
    
    #pred_mask = cv2.erode(pred,kernel,iterations = 1)

    pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
    pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)

    #pred_mask = cv2.dilate(pred_mask,kernel,iterations = 1)       
    #pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)  
    #pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
     
    pred = pred_mask        
    #r,c = pred_mask.shape       
    #pred_mask[0:1,:]=0
    #pred_mask[r-1:r,:]=0    
    #pred_mask[:,0:2]=0
    #pred_mask[:,c-2:c]=0 
    
    return pred_mask

def perform_morphological_operations_on_ROI(pred):
    
    #pred_mask = cv2.erode(pred,kernel,iterations = 1)

    pred_mask = cv2.dilate(pred,ROI_kernel,iterations = 1)            
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
    pred_mask = cv2.erode(pred_mask,ROI_kernel,iterations = 1)

    #pred_mask = cv2.dilate(pred_mask,kernel,iterations = 1)       
    #pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)  
    #pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
     
    pred = pred_mask        
    #r,c = pred_mask.shape       
    #pred_mask[0:1,:]=0
    #pred_mask[r-1:r,:]=0    
    #pred_mask[:,0:2]=0
    #pred_mask[:,c-2:c]=0 
    
    return pred_mask


def perform_dilation_operations(pred):
    #pred_mask = cv2.erode(pred,kernel,iterations = 1)
    pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
    return pred_mask

def perform_erosion_operations(pred):
    pred_mask = cv2.erode(pred,wsi_kernel,iterations = 1)
    #pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
    return pred_mask

def remove_small_tissue(bw_img, min_size=10000):
    """ Remove small holes in tissue image
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    min_size: int
        Minimum tissue area.
    Returns
    -------
    bw_remove: np.array
        Binary image with small tissue regions removed
    """

    bw_remove = remove_small_objects(bw_img, min_size=min_size, connectivity=8)

    return bw_remove


def find_tissue_cnts(bw_img):
    """ Fint contours of tissues
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    Returns
    -------
    cnts: list
        List of all contours coordinates of tissues.
    """

    _, cnts, _ = cv2.findContours(img_as_ubyte(bw_img),
                                  mode=cv2.RETR_EXTERNAL,
                                  method=cv2.CHAIN_APPROX_NONE)

    return cnts

def ROI_image_to_mask(slide_img, thresh_val, smooth_sigma, min_tissue_size):
    
    gray_img = rgb2gray(slide_img)
    # Step 4: Smooth and Binarize
    bw_img = thresh_slide(gray_img, thresh_val, sigma=smooth_sigma)
    # Step 5: Fill tissue holes
    bw_fill = fill_tissue_holes(bw_img)
    # Step 6: Remove small tissues
    bw_remove = remove_small_tissue(bw_fill, min_tissue_size)
    bw_remove = bw_remove.astype('uint8') * 255 
    
    #bimg_name =file_name_wo_ext+'_binary_img.jpg'
    #final_bimg_des = os.path.join(abspath,bimg_name)
    #imsave(final_bimg_des,bw_remove)
    #cv2.imwrite(final_bimg_des,bw_remove)
    
    return bw_remove

def extract_patches_from_binary_image(full_img,patch_h,patch_w, img_name, patches_saving_dir):

    height,width = full_img.shape    
    rows = (int) (height/patch_h)+1
    columns = (int) (width/patch_w) +1     
    

    mask_with_pad = np.zeros((rows*patch_h,columns*patch_w), dtype=np.uint8)
    #mask_with_pad = np.zeros(rows*patch_h,columns*patch_w)    
    mask_with_pad [0:height,0:width] = full_img
    
    image_id = img_name.split('_')[0]
    img_name_part1 = img_name.split('_')[1]
    img_name_part2 = img_name.split('_')[2]
    second_part_name = img_name_part1+'_'+img_name_part2
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = mask_with_pad[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
            #f_img_name =img_name+'_'+str(pn)+'.jpg'
            f_img_name =image_id+'_'+str(pn)+'_'+second_part_name+'.jpg'
            final_des_img = os.path.join(patches_saving_dir,f_img_name)            
            mx_val = patch_img.max()
            mn_val = patch_img.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))          
            cv2.imwrite(final_des_img,patch_img)
            #cv2.imwrite(final_des_mask,patch_mask)
            pn+=1
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def extract_patches_from_image(full_img,patch_h,patch_w, img_name, patches_saving_dir):

    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)+1
    columns = (int) (width/patch_w)+1           
    
    image_with_pad = np.zeros((rows*patch_h,columns*patch_w,3), dtype=np.uint8)
    #image_with_pad = np.zeros(rows*patch_h,columns*patch_w,3)    
    image_with_pad [0:height,0:width,:] = full_img
    image_id = img_name.split('_')[0]
    img_name_part1 = img_name.split('_')[1]
    img_name_part2 = img_name.split('_')[2]
    second_part_name = img_name_part1+'_'+img_name_part2
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = image_with_pad[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =image_id+'_'+str(pn)+'_'+second_part_name+'.jpg'
            final_des_img = os.path.join(patches_saving_dir,f_img_name)            
            mx_val = patch_img.max()
            mn_val = patch_img.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))          
            cv2.imwrite(final_des_img,patch_img)
            #cv2.imwrite(final_des_mask,patch_mask)
            pn+=1
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def extract_patches_from_image_with_ROI_mask_for_WSI(full_img,patch_h,patch_w, img_name, patches_saving_dir):

    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)+1
    columns = (int) (width/patch_w)+1           
    
    image_with_pad = np.zeros((rows*patch_h,columns*patch_w,3), dtype=np.uint8)
    #image_with_pad = np.zeros(rows*patch_h,columns*patch_w,3)    
    image_with_pad [0:height,0:width,:] = full_img
    image_id = img_name.split('_')[0]
    img_name_part1 = img_name.split('_')[1]
    img_name_part2 = img_name.split('_')[2]
    second_part_name = img_name_part1+'_'+img_name_part2
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = image_with_pad[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =image_id+'_'+str(pn)+'_'+second_part_name+'.jpg'
            final_des_img = os.path.join(patches_saving_dir,f_img_name)                    
            mx_val = patch_img.max()
            mn_val = patch_img.min()            
            print ('max_val : '+str(mx_val))
            print ('min_val : '+str(mn_val))    
            
            # create the maks for this input ROI image....
            copy_image = patch_img. copy()

            smooth_sigma=13
            thresh_val = 0.80,
            min_tissue_size=5000
            roi_mask_image = ROI_image_to_mask(copy_image, thresh_val, smooth_sigma, min_tissue_size)
            ## Refine the ROI for removing the border of the images...
            roi_mask_image_refined = 255.0*(roi_mask_image[:,:] >= 125)   
            #roi_mask_image_refined = perform_morphological_operations_on_ROI(roi_mask_image_refined)  
            #roi_mask_image_refined = fill_tissue_holes(roi_mask_image_refined)
            #(thresh, blackAndWhiteImage) = cv2.threshold(roi_mask_image_refined, 127, 255, cv2.THRESH_BINARY)
            #roi_mask_image_refined = 255.0*(roi_mask_image_refined[:,:] >= 125)   
            #roi_mask_image_refined = roi_mask_image_refined.astype('uint8') * 255 
            sf_part = second_part_name.split('_')[0]
            f_mask_name =image_id+'_'+str(pn)+'_'+sf_part+'_mask.jpg'
            final_img_des = os.path.join(patches_saving_dir,f_mask_name)
            cv2.imwrite(final_img_des,roi_mask_image_refined)
            #print('Total number of pixel before definement :',np.sum(roi_mask_image==255))
            #img_name =str(image_id)+'_actual_image_refined_mask.jpg'
            #final_img_des = os.path.join(patch_saving_dir,img_name)
            #cv2.imwrite(final_img_des,roi_mask_image_refined)
            #print('Total number of pixel after definement :',np.sum(roi_mask_image_refined==255))
            cv2.imwrite(final_des_img,patch_img)
            #cv2.imwrite(final_des_mask,patch_mask)
            pn+=1
        k +=1
        print ('Processing for: ' +str(k))
        
    return pn

def extract_same_size_patches_from_wsi(svs_img_dir, patches_saving_dir, patch_size):
    '''        
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
        
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
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
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
                cv2.imwrite(final_img_des,img)
                #imsave(final_img_des,img)                
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

def extract_same_size_patches_from_wsi_final(svs_img_dir, patches_saving_dir, patch_size):
    
           
#    patch_dir_name = 'patches_'+str(patch_size[0])+'/ 
#    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
#        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
#    patches_dir = join_path(patches_saving_dir+patch_dir_name)
#    
    patches_dir = patches_saving_dir
    patches_sub_dir = []
    
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
                
                idx_sr_sc_xy = str(img_saving_idx)+'_'+str(x)+'_'+str(y)                
                starting_row_columns.append(idx_sr_sc_xy)
                print("Processing : "+str(idx_sr_sc_xy))                
                ac_img_name =dir_name+'_'+str(idx_sr_sc_xy)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                cv2.imwrite(final_img_des,img)
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

def extract_same_size_patches_from_wsi_final_from_NASH(svs_img_dir, patches_saving_dir, patch_size, patches_exist=False):
        
    patches_dir = patches_saving_dir
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] in ['.tif' , '.svs'] ]
    
    for i, f in enumerate(image_svs):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            
            dir_name = os.path.splitext(f)[0]              
            print(svs_img_dir.split('/')[0])        
            if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
                os.makedirs("%s/%s"%(patches_dir,dir_name))
            
            patches_sub_dir = join_path(patches_dir,dir_name+'/')
            if not patches_exist:
                # open scan
                svs_img_path = os.path.join(svs_img_dir,f)
                scan = openslide.OpenSlide(svs_img_path)
                
                scan_dimensions = scan.dimensions        
                orig_w = scan_dimensions[0]
                orig_h = scan_dimensions[1]           
                no_patches_x_axis = math.ceil(orig_w/patch_size[0])
                no_patches_y_axis = math.ceil(orig_h/patch_size[1])                

                starting_row_columns = []
                img_saving_idx = 0        

                for y in range(0, orig_h, patch_size[1]):
                    for x in range(0, orig_w, patch_size[0]):                                 
                        img = np.array(scan.read_region((x,y),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]
                        idx_sr_sc = str(img_saving_idx)+','+str(x)+','+str(y)                
                        starting_row_columns.append(idx_sr_sc)
                        print("Processing:"+str(img_saving_idx))                
                        ac_img_name =dir_name+'_'+str(img_saving_idx)+'.jpg'
                        final_img_des = os.path.join(patches_sub_dir, ac_img_name)
                        print("Saving to:"+str(final_img_des))
                        im = Image.fromarray(img)
                        im.save(final_img_des)                  
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
        
    return os.path.dirname(patches_sub_dir), image_svs


def extract_same_size_patches_from_wsi_with_xy_final_PIL(svs_img_dir, patches_saving_dir, patch_size):
    
           
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


def extract_same_size_patches_from_wsi_final_one(svs_img_dir, patches_saving_dir, patch_size):
#    patch_dir_name = 'patches_'+str(patch_size[0])+'/ 
#    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
#        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
#    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    patches_dir = patches_saving_dir
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.tif']
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
                
                idx_sr_sc = str(img_saving_idx)+','+str(x)+','+str(y)                
                starting_row_columns.append(idx_sr_sc)
                print("Processing:"+str(img_saving_idx))                
                ac_img_name =dir_name+'_'+str(img_saving_idx)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                cv2.imwrite(final_img_des,img)
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


def extract_all_patches_from_svs(svs_img_dir, patches_saving_dir, patch_size):
            
#    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
#        
#    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
#        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
#    
#    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    patches_dir = patches_saving_dir
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs']
        
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_svs):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]              
        print(svs_img_dir.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        svs_img_path = os.path.join(svs_img_dir,f)
        scan = openslide.OpenSlide(svs_img_path)
        
        scan_dimensions = scan.dimensions
        
        orig_w = scan_dimensions[0]
        orig_h = scan_dimensions[1]
        #orig_w = np.int(scan.properties.get('aperio.OriginalWidth'))
        #orig_h = np.int(scan.properties.get('aperio.OriginalHeight'))               
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]                
        # create an array to store our image
        #img_np = np.zeros((orig_w,orig_h,3),dtype=np.uint8)        
        #pdb.set_trace()
        starting_row_columns = []

        img_saving_idx = 0
        
        for r in range(0,orig_h,patch_size[1]):
            for c in range(0, orig_w,patch_size[0]):
                
                if c+patch_size[1] > orig_w and r+patch_size[0]<= orig_h:
                    p = orig_w-c
                    img = np.array(scan.read_region((c,r),0,(p,patch_size[1])),dtype=np.uint8)[...,0:3]
                elif c+patch_size[1] <= orig_w and r+patch_size[0] > orig_h:
                    p = orig_h-r
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],p)),dtype=np.uint8)[...,0:3]
                elif  c+patch_size[1] > orig_w and r+patch_size[0] > orig_h:
                    p = orig_h-c
                    pp = orig_w-r
                    img = np.array(scan.read_region((c,r),0,(p,pp)),dtype=np.uint8)[...,0:3]
                else:    
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]
                 
                idx_sr_sc = str(img_saving_idx)+','+str(c)+','+str(r)                
                starting_row_columns.append(idx_sr_sc)
                print("Processing:"+str(img_saving_idx))                
                ac_img_name =str(img_saving_idx)+'.jpg'
                final_img_des = os.path.join(patches_sub_dir,ac_img_name)
                cv2.imwrite(final_img_des,img)
                #imsave(final_img_des,img)   
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
    

def patches_to_image(patches_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    
    names_wo_ext = []
    for idx in range(len(image_files)):
        name = image_files[idx]
        name_wo_ext = int(name.split('.')[0])
        names_wo_ext.append(name_wo_ext)
    
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = image_logs["height"]
        image_w = image_logs["width"]
        patch_w = image_logs["patch_width"]
        patch_h = image_logs["patch_height"]        
        num_rows = image_logs["no_patches_x_axis"]
        num_columns = image_logs["no_patches_y_axis"]
            
    img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8)

    
    #pdb.set_trace()     
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            name = str(patches_name_wo_ext[patch_idx])+'.jpg'
            patch = cv2.imread(patches_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(name)
            print(patch.shape)
            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    img_name =str(image_id)+'_merge.jpg'
    final_img_des = os.path.join(patches_dir,img_name)
    cv2.imwrite(final_img_des,img_from_patches)
    
    
def patch2subpatches_driver(patches_source, patches_saving_dir,patch_size):
    
    #pdb.set_trace()
    
    dir_name = "patches_"+str(patch_size[0])
    if not os.path.isdir("%s/%s"%(patches_saving_dir,dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,dir_name))
    
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'    
    patches_dir = join_path(patches_saving_dir,patch_dir_name)    
    image_dirs = [x for x in sorted(os.listdir(patches_source)) if x[-4:] == '.jpg']
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_dirs):
        
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]               
        img_name = f
        print(patches_source.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        img_path = os.path.join(patches_source,f)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        
        orig_w,orig_h,channels = img.shape
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]
                
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        patches_number = extract_patches_from_image(img,patch_size[0],patch_size[1], img_name, patches_sub_dir)
   
        print(str(patches_number))
        
        
def patches_to_image_heatmaps(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-29:] == '_image_seg_class_heatmaps.jpg']
    
    #pdb.set_trace()

    names_wo_ext = []
#    for idx in range(len(image_files)):
#        name = image_files[idx]        
#        name_wo_ext_part1 = name.split('.')[0]
#        text_part  = name_wo_ext_part1.split('_')[1]+'_'+name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3]+'_'+name_wo_ext_part1.split('_')[4]
#        name_wo_ext_num = int(name_wo_ext_part1.split('_')[0])
#        names_wo_ext.append(name_wo_ext_num)
        
#    for idx in range(len(image_files)):
#        name = image_files[idx]        
#        name_wo_ext_part1 = name.split('.')[0]
#        text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
#        text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3]+'_'+name_wo_ext_part1.split('_')[4] #'_'+name_wo_ext_part1.split('_')[3]
#        name_wo_ext_num = int(name_wo_ext_part1.split('_')[1])
#        names_wo_ext.append(name_wo_ext_num)
#        
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3]+'_'+name_wo_ext_part1.split('_')[4]+'_'+name_wo_ext_part1.split('_')[5] #'_'
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    #pdb.set_trace()
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])
        num_columns = int(image_logs["no_patches_x_axis"])
            
    #img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8) # original
    img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED)#.astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            #print(img_name)
            #print(patch.shape)            
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    #dim = (5096,4096)    
    dim = (10240,8192)   
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_heatmaps.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)   
    
    patch_h = 10240
    patch_w = 8192
    #pdb.set_trace() 
    img_name = str(image_id)+'_heatmaps'
    extract_patches_from_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)


#def save_big_patches_for_wsi(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir):
    
    

def patches_to_actual_image(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-15:] == '_actual_img.jpg']
    
#    names_wo_ext = []
#    for idx in range(len(image_files)):
#        name = image_files[idx]        
#        name_wo_ext_part1 = name.split('.')[0]
#        text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
#        text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
#        name_wo_ext_num = int(name_wo_ext_part1.split('_')[1])
#        names_wo_ext.append(name_wo_ext_num)
    
    name = image_files[0] 
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
   #extension = '.jpg'    
    #patches_name_wo_ext = np.array(names_wo_ext)       
    #patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    #pdb.set_trace()
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])
        num_columns = int(image_logs["no_patches_x_axis"])
            
    #img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8) # original
    img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            #img_name = text_part_first+'_'+str(patches_name_wo_ext[patch_idx])+'_'+text_part_last+'.jpg'
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(img_name)
            print(patch.shape)
            
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original
            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch

            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1

    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_actual_image.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    #pdb.set_trace()  
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_img' 
    extract_patches_from_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    

def patches_to_binary_image(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-14:] == '_image_seg.jpg']
    

    #pdb.set_trace()
    names_wo_ext = []
#    for idx in range(len(image_files)):
#        name = image_files[idx]        
#        name_wo_ext_part1 = name.split('.')[0]
#        text_part  = name_wo_ext_part1.split('_')[1]+'_'+name_wo_ext_part1.split('_')[2]  #+'_'+name_wo_ext_part1.split('_')[3]
#        name_wo_ext_num = int(name_wo_ext_part1.split('_')[0])
#        names_wo_ext.append(name_wo_ext_num)
#    
#    for idx in range(len(image_files)):
#        name = image_files[idx]        
#        name_wo_ext_part1 = name.split('.')[0]
#        text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
#        text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
#        name_wo_ext_num = int(name_wo_ext_part1.split('_')[1])
#        names_wo_ext.append(name_wo_ext_num)

    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
        
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    #pdb.set_trace()
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])
        num_columns = int(image_logs["no_patches_x_axis"])
            
    #img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8) # original
    img_from_patches = np.zeros((image_h,image_w),dtype=np.uint8)

    
    #pdb.set_trace()  
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(img_name)
            print(patch.shape)
            
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original            
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w] = patch
        
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_mask.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)
    
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_mask' 
    extract_patches_from_binary_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    


def patches_to_image_v2(patches_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    
    names_wo_ext = []
    for idx in range(len(image_files)):
        name = image_files[idx]
        name_wo_ext = int(name.split('.')[0])
        names_wo_ext.append(name_wo_ext)
    
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = image_logs["height"]
        image_w = image_logs["width"]
        patch_w = image_logs["patch_width"]
        patch_h = image_logs["patch_height"]        
        num_rows = image_logs["no_patches_x_axis"]
        num_columns = image_logs["no_patches_y_axis"]
            
    img_from_patches = np.zeros((image_w,image_h,3),dtype=np.uint8)

    
    #pdb.set_trace()     
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            name = str(patches_name_wo_ext[patch_idx])+'.jpg'
            patch = cv2.imread(patches_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(name)
            print(patch.shape)
            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    img_name =str(image_id)+'_merge.jpg'
    final_img_des = os.path.join(patches_dir,img_name)
    #imsave(final_img_des,img_from_patches)
    cv2.imwrite(final_img_des,img_from_patches)
    
def patch2subpatches_driver_v2(patches_source, patches_saving_dir,patch_size):
    
    #pdb.set_trace()
    
    dir_name = "patches_"+str(patch_size[0])
    if not os.path.isdir("%s/%s"%(patches_saving_dir,dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,dir_name))
    
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'    
    patches_dir = join_path(patches_saving_dir,patch_dir_name)    
    image_dirs = [x for x in sorted(os.listdir(patches_source)) if x[-4:] == '.jpg']
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_dirs):
        
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]               
        img_name = f
        print(patches_source.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        img_path = os.path.join(patches_source,f)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        
        orig_w,orig_h,channels = img.shape
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]
                
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        patches_number = extract_patches_from_image(img,patch_size[0],patch_size[1], img_name, patches_sub_dir)
   
        print(str(patches_number))
        
 
    
        
def patches_to_image_heatmaps_medpace(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-29:] == '_image_seg_class_heatmaps.jpg']
    
    #pdb.set_trace()
    names_wo_ext = []
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3]+'_'+name_wo_ext_part1.split('_')[4]+'_'+name_wo_ext_part1.split('_')[5] #'_'
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    #pdb.set_trace()
 
    json_path = join_path(patches_dir,json_files[0])
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1

    #pdb.set_trace()         
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED)#.astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            #print(img_name)
            #print(patch.shape)            
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    #dim = (5096,4096)    
    dim = (10240,8192)   
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_heatmaps_img.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)   
    
#    patch_h = 10240
#    patch_w = 8192
#    #pdb.set_trace() 
#    img_name = str(image_id)+'_heatmaps'
#    extract_patches_from_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)

def patches_to_actual_image_medpace(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-15:] == '_actual_img.jpg'] 

    name = image_files[0] 
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          
        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
    
    
    #pdb.set_trace()         
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            #img_name = text_part_first+'_'+str(patches_name_wo_ext[patch_idx])+'_'+text_part_last+'.jpg'
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            print(img_name)
            #print(patch.shape)
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
            
        #pdb.set_trace()
        print('Row : '+str(r)+ 'and columns :'+str(c))

    #pdb.set_trace() 
  
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_actual_image.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

#    pdb.set_trace()
#    
#    smooth_sigma=13
#    thresh_val = 0.80,
#    min_tissue_size=10000
#                       
#    roi_mask_image = ROI_image_to_mask(resized_img, thresh_val, smooth_sigma, min_tissue_size)
#
#    img_name =str(image_id)+'_actual_image_mask.jpg'
#    final_img_des = os.path.join(patch_saving_dir,img_name)
#    cv2.imwrite(final_img_des,roi_mask_image)
    
    #pdb.set_trace()  
    #patch_h = 10240
    #patch_w = 8192
    #img_name = str(image_id)+'_img' 
    #extract_patches_from_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
def patches_to_actual_image_and_mask_medpace(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-15:] == '_actual_img.jpg'] 

    name = image_files[0] 
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          
        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
    
    
    #pdb.set_trace()         
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            #img_name = text_part_first+'_'+str(patches_name_wo_ext[patch_idx])+'_'+text_part_last+'.jpg'
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            print(img_name)
            #print(patch.shape)
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
            
        #pdb.set_trace()
        print('Row : '+str(r)+ 'and columns :'+str(c))

    #pdb.set_trace() 
  
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_actual_image.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    #pdb.set_trace()
    
    smooth_sigma=13
    thresh_val = 0.80,
    min_tissue_size=1000
                       
    roi_mask_image = ROI_image_to_mask(resized_img, thresh_val, smooth_sigma, min_tissue_size)

    img_name =str(image_id)+'_actual_image_mask.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,roi_mask_image)
    

  
    
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_ac_img' 
    extract_patches_from_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
    #pdb.set_trace()  
    dim = (img_from_patches.shape[1], img_from_patches.shape[0])
    resized_mask = cv2.resize(roi_mask_image,dim) 
    img_name = str(image_id)+'_ac_mask' 
    extract_patches_from_binary_image(resized_mask,patch_h,patch_w, img_name, patch_saving_dir)

def patches_to_actual_image_and_refined_mask_medpace(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-15:] == '_actual_img.jpg'] 

    name = image_files[0] 
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          
        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
    
    
    #pdb.set_trace()         
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            #img_name = text_part_first+'_'+str(patches_name_wo_ext[patch_idx])+'_'+text_part_last+'.jpg'
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            print(img_name)
            #print(patch.shape)
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
            
        #pdb.set_trace()
        print('Row : '+str(r)+ 'and columns :'+str(c))

    #pdb.set_trace() 
  
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_actual_image.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    #pdb.set_trace()
    
    smooth_sigma=13
    thresh_val = 0.80,
    min_tissue_size=1000
                       
    roi_mask_image = ROI_image_to_mask(resized_img, thresh_val, smooth_sigma, min_tissue_size)
    
    img_name =str(image_id)+'_actual_image_mask.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,roi_mask_image)
    print('Total number of pixel before definement :',np.sum(roi_mask_image==255))
    ## Refine the ROI for removing the border of the images...
    roi_mask_image_refined = 255.0*(roi_mask_image[:,:] >= 125)   
    roi_mask_image_refined = perform_erosion_operations(roi_mask_image_refined)  
    img_name =str(image_id)+'_actual_image_refined_mask.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,roi_mask_image_refined)
    print('Total number of pixel after definement :',np.sum(roi_mask_image_refined==255))

    #pdb.set_trace()
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_ac_img' 
    extract_patches_from_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
    #pdb.set_trace()  
    dim = (img_from_patches.shape[1], img_from_patches.shape[0])
    #resized_mask = cv2.resize(roi_mask_image,dim) ### result without refined mask
    resized_mask = cv2.resize(roi_mask_image_refined,dim) ### result with refined mask
    
    img_name = str(image_id)+'_ac_mask' 
    extract_patches_from_binary_image(resized_mask,patch_h,patch_w, img_name, patch_saving_dir)


def patches_to_actual_image_and_ROI_refined_mask_medpace(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-15:] == '_actual_img.jpg'] 

    name = image_files[0] 
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          
        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
    
    
    #pdb.set_trace()         
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            #img_name = text_part_first+'_'+str(patches_name_wo_ext[patch_idx])+'_'+text_part_last+'.jpg'
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            print(img_name)
            #print(patch.shape)
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
            
        #pdb.set_trace()
        print('Row : '+str(r)+ 'and columns :'+str(c))

    #pdb.set_trace() 
  
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_actual_image.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    #pdb.set_trace()
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_ac_img' 
    extract_patches_from_image_with_ROI_mask_for_WSI(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
#    #pdb.set_trace()  
#    dim = (img_from_patches.shape[1], img_from_patches.shape[0])
#    #resized_mask = cv2.resize(roi_mask_image,dim) ### result without refined mask
#    resized_mask = cv2.resize(roi_mask_image_refined,dim) ### result with refined mask
#    
#    img_name = str(image_id)+'_ac_mask' 
#    extract_patches_from_binary_image(resized_mask,patch_h,patch_w, img_name, patch_saving_dir)



#def refine_binarymask_and_heatmaps(patches_dir,patch_saving_dir):
    
    
def patches_to_binary_image_medpace(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-14:] == '_image_seg.jpg']
    

    #pdb.set_trace()
    names_wo_ext = []
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] #'_'+name_wo_ext_part1.split('_')[3]
        
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
       
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
           
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final),dtype=np.uint8)
    #pdb.set_trace()  
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(img_name)
            print(patch.shape)

            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w] = patch
        
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_seg_mask.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)
    
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_seg_mask' 
    extract_patches_from_binary_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
def patches_to_binary_image_from_seg_class_masks_medpace_final(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    #image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-14:] == '_image_seg.jpg']
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-20:] == '_image_seg_class.jpg']


    #pdb.set_trace()
    names_wo_ext = []
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    ## if you don't use the 4th part it will read only the seg masks......
    text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] +'_'+name_wo_ext_part1.split('_')[4] 
        
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
       
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
           
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final),dtype=np.uint8)
    
    #pdb.set_trace()  
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = text_part_first+'_'+str(patch_idx)+'_'+text_part_last+'.jpg'
            patch = cv2.imread(patches_dir + img_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(img_name)
            print(patch.shape)

            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w] = patch
        
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    #pdb.set_trace()
    
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_seg_mask.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_seg_mask' 
    extract_patches_from_binary_image(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
    
def create_heatmaps_from_roi_images_from_class_plus_seg_mask(roi_imagelogs_dir,patch_saving_dir):
    
    image_files = [x for x in sorted(os.listdir(roi_imagelogs_dir)) if x[-11:] == '_ac_img.jpg']
    
    #pdb.set_trace()
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    actual_wsi_name  = name_wo_ext_part1.split('_')[0]
   

    num_rois = len(image_files)
    
    for k in range(num_rois):
        idv_roi_name = image_files[k]
        idv_name_wo_ext_part1 = idv_roi_name.split('.')[0]
        print('Processing for the ROI id of:',idv_name_wo_ext_part1)

        #idvactual_wsi_name  = idv_name_wo_ext_part1.split('_')[0]
        idv_roi_id = idv_name_wo_ext_part1.split('_')[1]
        #roi_id = int(name_wo_ext_part1.split('_')[-1])
        actual_roi_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_ac_mask.jpg'
        seg_roi_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_seg_mask.jpg'
        seg_mask_path = os.path.join(roi_imagelogs_dir,seg_roi_mask_name)
        roi_image = cv2.imread(roi_imagelogs_dir + idv_roi_name, cv2.IMREAD_UNCHANGED)
        roi_mask_input = cv2.imread(roi_imagelogs_dir + actual_roi_mask_name)
        roi_seg_mask_input = cv2.imread(roi_imagelogs_dir + seg_roi_mask_name)

        gray_mask_input = cv2.cvtColor(roi_mask_input, cv2.COLOR_BGR2GRAY)
        gray_seg_mask_input = cv2.cvtColor(roi_seg_mask_input, cv2.COLOR_BGR2GRAY)

        #roi_mask = 255.0*(gray_mask_input[:,:] >= 125)   
        #roi_mask = perform_erosion_operations(roi_mask)  
        roi_mask = 255.0*(gray_mask_input[:,:] >= 125)             
        roi_seg_mask = 255.0*(gray_seg_mask_input[:,:] >= 125)               

        # perform AND operation between seg-mask and actual-image mask
#        combined_seg_mask = cv2.bitwise_and(roi_mask, roi_seg_mask, mask = None) 
#        seg_roi_combined_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_combined_mask.jpg'
#        final_des_combined_mask = os.path.join(patch_saving_dir,seg_roi_combined_mask_name)
#        cv2.imwrite(final_des_combined_mask,combined_seg_mask)
#        
#        
#        #pdb.set_trace()
#        #pred_masks_seg = 255.0*(combined_seg_mask[:,:,:] >= 0.5)               
#        reconstructed_combined_seg_morph = perform_dilation_operations(combined_seg_mask) 
#        seg_roi_combined_with_morphology_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_combined_wmorph_mask.jpg'
#        final_des_combined_mask_morph = os.path.join(patch_saving_dir,seg_roi_combined_with_morphology_mask_name)
#        cv2.imwrite(final_des_combined_mask_morph,reconstructed_combined_seg_morph)
        
        reconstructed_pred_seg_morph =cv2.imread(seg_mask_path) ## heatmap from the seg_class mask...
        #reconstructed_pred_seg_morph =cv2.imread(final_des_combined_mask_morph) ## heatmap from the combined mask...
        im_heatmap = create_heatmap(roi_image, reconstructed_pred_seg_morph, a1=.5, a2=.5)
        
        y_pred_name_seg_heatmap = actual_wsi_name+'_'+str(idv_roi_id)+'_heatmaps'+'.jpg' 
        final_des_pred_seg_heatmap = os.path.join(patch_saving_dir,y_pred_name_seg_heatmap)
        cv2.imwrite(final_des_pred_seg_heatmap,im_heatmap)
        
        #pdb.set_trace()
        total_pixels = roi_image.shape[0]*roi_image.shape[1]
        total_steatosis_pixels = np.sum(roi_seg_mask == 255)  
        total_number_tissue_pixels = np.sum(roi_mask == 255) 
        
        #pixels_wo_steatosis = total_number_tissue_pixels - total_steatosis_pixels
        percentage_of_steatosis_cells = (total_steatosis_pixels/total_number_tissue_pixels)*100
        
        if percentage_of_steatosis_cells == 'nan':
            percentage_of_steatosis_cells = 0
            
        print('The percentage of steatosis pixels :', percentage_of_steatosis_cells)
        ## Creat text file to save logs for the input regions....
        log_file_name_for_roi = actual_wsi_name+'_'+str(idv_roi_id)+'_logs.txt'
        roi_output_logs = join_path(patch_saving_dir,log_file_name_for_roi)        
        patch_logs = open(roi_output_logs, 'w')
         
        patch_logs.write("Total_pixels in ROI : "+str(total_pixels)
                            + "\n Total steatosis cells in ROI: " +str(total_steatosis_pixels)
                            + "\n Total tissue pixels in ROI:"+str(total_number_tissue_pixels)
                            + "\n Steatosis in percentage in ROI:"+str(percentage_of_steatosis_cells)
                            )

      
def create_heatmaps_from_roi_images_from_combined_seg_mask(roi_imagelogs_dir,patch_saving_dir):
    
    image_files = [x for x in sorted(os.listdir(roi_imagelogs_dir)) if x[-11:] == '_ac_img.jpg']
    
    #pdb.set_trace()
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    actual_wsi_name  = name_wo_ext_part1.split('_')[0]
   

    num_rois = len(image_files)
    
    for k in range(num_rois):
        idv_roi_name = image_files[k]
        idv_name_wo_ext_part1 = idv_roi_name.split('.')[0]
        print('Processing for the ROI id of:',idv_name_wo_ext_part1)

        #idvactual_wsi_name  = idv_name_wo_ext_part1.split('_')[0]
        idv_roi_id = idv_name_wo_ext_part1.split('_')[1]
        #roi_id = int(name_wo_ext_part1.split('_')[-1])
        actual_roi_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_ac_mask.jpg'
        seg_roi_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_seg_mask.jpg'
        #seg_mask_path = os.path.join(roi_imagelogs_dir,seg_roi_mask_name)
        roi_image = cv2.imread(roi_imagelogs_dir + idv_roi_name, cv2.IMREAD_UNCHANGED)
        roi_mask_input = cv2.imread(roi_imagelogs_dir + actual_roi_mask_name)
        roi_seg_mask_input = cv2.imread(roi_imagelogs_dir + seg_roi_mask_name)

        gray_mask_input = cv2.cvtColor(roi_mask_input, cv2.COLOR_BGR2GRAY)
        gray_seg_mask_input = cv2.cvtColor(roi_seg_mask_input, cv2.COLOR_BGR2GRAY)

        #roi_mask = 255.0*(gray_mask_input[:,:] >= 125)   
        #roi_mask = perform_erosion_operations(roi_mask)  
        roi_mask = 255.0*(gray_mask_input[:,:] >= 125)             
        roi_seg_mask = 255.0*(gray_seg_mask_input[:,:] >= 125)               

        # perform AND operation between seg-mask and actual-image mask
        combined_seg_mask = cv2.bitwise_and(roi_mask, roi_seg_mask, mask = None) 
        seg_roi_combined_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_combined_mask.jpg'
        final_des_combined_mask = os.path.join(patch_saving_dir,seg_roi_combined_mask_name)
        cv2.imwrite(final_des_combined_mask,combined_seg_mask)
        
        
        #pdb.set_trace()
        #pred_masks_seg = 255.0*(combined_seg_mask[:,:,:] >= 0.5)               
        reconstructed_combined_seg_morph = perform_dilation_operations(combined_seg_mask) 
        seg_roi_combined_with_morphology_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_combined_wmorph_mask.jpg'
        final_des_combined_mask_morph = os.path.join(patch_saving_dir,seg_roi_combined_with_morphology_mask_name)
        cv2.imwrite(final_des_combined_mask_morph,reconstructed_combined_seg_morph)
        
        #reconstructed_pred_seg_morph =cv2.imread(seg_mask_path) ## heatmap from the seg_class mask...
        reconstructed_pred_seg_morph =cv2.imread(final_des_combined_mask_morph) ## heatmap from the combined mask...
        im_heatmap = create_heatmap(roi_image, reconstructed_pred_seg_morph, a1=.5, a2=.5)
        
        y_pred_name_seg_heatmap = actual_wsi_name+'_'+str(idv_roi_id)+'_heatmaps'+'.jpg' 
        final_des_pred_seg_heatmap = os.path.join(patch_saving_dir,y_pred_name_seg_heatmap)
        cv2.imwrite(final_des_pred_seg_heatmap,im_heatmap)
        
        #pdb.set_trace()
        
        total_pixels = roi_image.shape[0]*roi_image.shape[1]
        total_steatosis_pixels = np.sum(reconstructed_combined_seg_morph == 255)  
        total_number_tissue_pixels = np.sum(roi_mask == 255) 
        
        #pixels_wo_steatosis = total_number_tissue_pixels - total_steatosis_pixels
        percentage_of_steatosis_cells = (total_steatosis_pixels/total_number_tissue_pixels)*100
        
        if percentage_of_steatosis_cells == 'nan':
            percentage_of_steatosis_cells = 0
            
        print('The percentage of steatosis pixels :', percentage_of_steatosis_cells)
        ## Creat text file to save logs for the input regions....
        log_file_name_for_roi = actual_wsi_name+'_'+str(idv_roi_id)+'_logs.txt'
        roi_output_logs = join_path(patch_saving_dir,log_file_name_for_roi)        
        patch_logs = open(roi_output_logs, 'w')
         
        patch_logs.write("Total_pixels in ROI : "+str(total_pixels)
                            + "\n Total steatosis cells in ROI: " +str(total_steatosis_pixels)
                            + "\n Total tissue pixels in ROI:"+str(total_number_tissue_pixels)
                            + "\n Steatosis in percentage in ROI:"+str(percentage_of_steatosis_cells)
                            )
        
def patches_to_image_medpace(patches_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    
   
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg']
    
    
    names_wo_ext = []
    for idx in range(len(image_files)):
        name = image_files[idx]
        name_wo_ext = int(name.split('.')[0])
        names_wo_ext.append(name_wo_ext)
    
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(names_wo_ext)       
    patches_name_wo_ext.sort()
    #patches_name_wo_ext = patches_name_wo_ext.tolist()
    
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :

         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = image_logs["height"]
        image_w = image_logs["width"]
        patch_w = image_logs["patch_width"]
        patch_h = image_logs["patch_height"]        
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
    
    
    #pdb.set_trace()         
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)

    
    #pdb.set_trace()     
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            name = str(patches_name_wo_ext[patch_idx])+'.jpg'
            patch = cv2.imread(patches_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(name)
            print(patch.shape)
            
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    img_name =str(image_id)+'_merge.jpg'
    final_img_des = os.path.join(patches_dir,img_name)
    #imsave(final_img_des,img_from_patches)
    cv2.imwrite(final_img_des,img_from_patches)
    
def patch2subpatches_driver_medpace(patches_source, patches_saving_dir,patch_size):
    
    #pdb.set_trace()
    
    dir_name = "patches_"+str(patch_size[0])
    if not os.path.isdir("%s/%s"%(patches_saving_dir,dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,dir_name))
    
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'    
    patches_dir = join_path(patches_saving_dir,patch_dir_name)    
    image_dirs = [x for x in sorted(os.listdir(patches_source)) if x[-4:] == '.jpg']
    #for f in os.listdir(image_svs):
    for i, f in enumerate(image_dirs):
        
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        
        dir_name = os.path.splitext(f)[0]               
        img_name = f
        print(patches_source.split('/')[0])        
        if not os.path.isdir("%s/%s"%(patches_dir,dir_name)):
            os.makedirs("%s/%s"%(patches_dir,dir_name))
        
        patches_sub_dir = join_path(patches_dir+dir_name+'/')
        # open scan
        img_path = os.path.join(patches_source,f)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        
        orig_w,orig_h,channels = img.shape
        no_patches_x_axis = orig_w/patch_size[0]
        no_patches_y_axis = orig_h/patch_size[1]
                
        svs_log = {}
        svs_log["ID"] = dir_name
        svs_log["height"] = orig_h
        svs_log["width"] = orig_w
        svs_log["patch_width"] = patch_size[0]
        svs_log["patch_height"] = patch_size[1]
        svs_log["no_patches_x_axis"] = no_patches_x_axis
        svs_log["no_patches_y_axis"] = no_patches_y_axis
        
        # make experimental log saving path...
        json_file = os.path.join(patches_sub_dir,'image_log.json')
        with open(json_file, 'w') as file_path:
            json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
        patches_number = extract_patches_from_image(img,patch_size[0],patch_size[1], img_name, patches_sub_dir)
   
        print(str(patches_number))
        
 

 ################### Function for fibrosis detection with new samples....

def patches_to_actual_image_and_ROI_refined_mask_medpace_fibrosis(patches_dir,patch_saving_dir):
    
    image_id = []
    image_h = []
    image_w = []
    patch_h = []
    patch_w = []
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-12:] == '_act_img.jpg'] 

    name = image_files[0]
    filename, file_extension = os.path.splitext(name)
    
    #name_wo_ext_part1 = name.split('.')[0]
    filename_wo_ID  = filename.split('_0_act_img')[0]
    #text_part_first  = text_part_first_with_ID.split('TRI')[0]#'_'+name_wo_ext_part1.split('_')[3]
    #text_part_last  =  name_wo_ext_part1.split('_')[4]+'_'+name_wo_ext_part1.split('_')[5] #'_'+name_wo_ext_part1.split('_')[3]
        
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          
        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
    
       
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final,3),dtype=np.uint8)
    
    #pdb.set_trace()      
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            #img_name = text_part_first+'_'+str(patches_name_wo_ext[patch_idx])+'_'+text_part_last+'.jpg'
            img_name = filename_wo_ID+'_'+str(patch_idx)+'_act_img.jpg'
            patch = cv2.imread(join_path(patches_dir,img_name), cv2.IMREAD_UNCHANGED).astype('float32')
            #img_from_patches[r,c]=patch  
            #print(patch.mean())
            print(img_name)
            #print(patch.shape)
            #img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch # original
            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w,:] = patch
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
            
        #pdb.set_trace()
        print('Row : '+str(r)+ 'and columns :'+str(c))

    #pdb.set_trace() 
  
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_resized_WSI.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    #pdb.set_trace()
    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_WSI' 
    extract_patches_from_image_with_ROI_mask_for_WSI(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)
    
#    #pdb.set_trace()  
#    dim = (img_from_patches.shape[1], img_from_patches.shape[0])
#    #resized_mask = cv2.resize(roi_mask_image,dim) ### result without refined mask
#    resized_mask = cv2.resize(roi_mask_image_refined,dim) ### result with refined mask
#    
#    img_name = str(image_id)+'_ac_mask' 
#    extract_patches_from_binary_image(resized_mask,patch_h,patch_w, img_name, patch_saving_dir)


def patches_to_binary_image_from_seg_class(patches_dir,patch_saving_dir, tissue_string,json_filename_token, des_filename_token):

    # json_filename_token = '.json'
    # des_filename_token = '_image_pred_seg_pp.jpg'
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-len(json_filename_token):] == json_filename_token]           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-len(des_filename_token):] == des_filename_token]

    name = image_files[0]        
    # name_wo_ext_part1 = name.split('.')[0]
    text_part_first  = name.split(des_filename_token)[0]#'_'+name_wo_ext_part1.split('_')[3]
    text_part_first = text_part_first.split('_')
    string_first_part = text_part_first[0]
    # For loop to '_' split in case of having several in a filename
    for t in range(1, len(text_part_first) - 1):
        string_first_part = string_first_part + '_' + text_part_first[t]

    json_path = join_path(patches_dir, json_files[0])

    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])    
        num_rows = math.ceil(image_logs["no_patches_y_axis"])
        num_columns = math.ceil(image_logs["no_patches_x_axis"])

    img_names = []
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = string_first_part+'_'+str(patch_idx)+des_filename_token
            if (c == num_columns - 1):
                patch_filename = join_path(patches_dir, img_name)
                patch_filename = os.path.normpath(patch_filename)
                patch = Image.open(patch_filename)
                patch_array = np.asarray(patch)  
                patch_array = patch_array[ : , 0: abs(c*patch_w - image_w)]
                Image.fromarray(patch_array).save(patch_filename)
           
            if (r == num_rows - 1):
                patch_filename = join_path(patches_dir, img_name)
                patch_filename = os.path.normpath(patch_filename)
                patch = Image.open(patch_filename)
                patch_array = np.asarray(patch) 
                patch_array = patch_array[ 0: abs(r*patch_h - image_h), : ]
                Image.fromarray(patch_array).save(patch_filename)
                
            img_names.append(join_path(patches_dir, img_name))
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    images = [pyvips.Image.new_from_file(filename, access="sequential") for filename in img_names]
    final = pyvips.Image.arrayjoin(images, across = num_columns)
    
    image_format = '.tif'
    img_name =str(image_id)+'_seg_mask_fb'+image_format
    final_img_des = os.path.join(patch_saving_dir,img_name)

    final_cropped = final.extract_area(0, 0, image_w, image_h)
    final_cropped.write_to_file(final_img_des, tile=True, compression="jpeg")
    
    print(f'mask filename is: {final_img_des}')
    # Convert the mask from image to svg file
    annotation_file_path = convert_mask_to_svg(tissue_string, final_img_des, closed_shape=True)
    print(annotation_file_path)
    # Compress the mask image to save space
    zipfile_path = zip(final_img_des)
    print(f'{final_img_des} is zipped to {zipfile_path}')

def patches_to_binary_mask_from_seg_fibrosis_medpace_final(patches_dir,patch_saving_dir):
    
    image_id = []
    patch_h = []
    patch_w = []
    
    json_files = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']           
    image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-12:] == '_wpp_seg.jpg']
    #image_files = [x for x in sorted(os.listdir(patches_dir)) if x[-19:] == '_image_seg.jpg']


    #pdb.set_trace()
    #names_wo_ext = []
    #name = image_files[0]        
    #name_wo_ext_part1 = name.split('.')[0]
    #text_part_first  = name_wo_ext_part1.split('_')[0]#'_'+name_wo_ext_part1.split('_')[3]
    ## if you don't use the 4th part it will read only the seg masks......
    #text_part_last  = name_wo_ext_part1.split('_')[2]+'_'+name_wo_ext_part1.split('_')[3] +'_'+name_wo_ext_part1.split('_')[4]+'_'+name_wo_ext_part1.split('_')[5] 
    
    name = image_files[0] 
    filename, file_extension = os.path.splitext(name)
    filename_wo_ID  = filename.split('_0_wpp_seg')[0]

#    name_wo_ext_part1 = name.split('.')[0]
#    text_part_first_with_ID  = name_wo_ext_part1.split('_image_seg')[0]
#    text_part_first  = text_part_first_with_ID.split('TRI')[0]#'_'+name_wo_ext_part1.split('_')[3]
#    text_part_last  =  name_wo_ext_part1.split('_')[4]+'_'+name_wo_ext_part1.split('_')[5] #'_'+name_wo_ext_part1.split('_')[3]
            
   #extension = '.jpg'    
    patches_name_wo_ext = np.array(image_files)       
    patches_name_wo_ext.sort()
       
    json_path = join_path(patches_dir,json_files[0])
    
    if len(json_files)<= 0 :
         print("The json file is not available")
    else:
        with open(json_path, "r") as f:
            image_logs = json.load(f)          

        image_id = image_logs["ID"]
        image_h = int(image_logs["height"])
        image_w = int(image_logs["width"])
        patch_w = int(image_logs["patch_width"])
        patch_h = int(image_logs["patch_height"])       
        num_rows = int(image_logs["no_patches_y_axis"])+1
        num_columns = int(image_logs["no_patches_x_axis"])+1
           
    #img_from_patches = np.zeros((image_h,image_w,3),dtype=np.uint8) # original
    image_h_final = int(num_rows*patch_h)
    image_w_final = int(num_columns*patch_w)
    img_from_patches = np.zeros((image_h_final,image_w_final),dtype=np.uint8)
    
    #pdb.set_trace()  
    
    patch_idx = 0        
    for r in range(0,num_rows):
        for c in range(0, num_columns):
            img_name = filename_wo_ID+'_'+str(patch_idx)+'_wpp_seg.jpg'
            #img_name = text_part_first+'TRI_'+str(patch_idx)+'_'+text_part_last+'.jpg'


            patch = cv2.imread(join_path(patches_dir,img_name), cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            #img_from_patches[r,c]=patch  
            print(patch.mean())
            print(img_name)
            print(patch.shape)

            img_from_patches[r*patch_h: r*patch_h+patch_h,c*patch_w:c*patch_w+patch_w] = patch
        
            print("Merging patch no. :"+str(patch_idx))               
            patch_idx +=1
    
    #pdb.set_trace()
    
    dim = (10240,8192)
    resized_img = cv2.resize(img_from_patches, dim, interpolation = cv2.INTER_AREA) 
    img_name =str(image_id)+'_seg_mask_WSI.jpg'
    final_img_des = os.path.join(patch_saving_dir,img_name)
    cv2.imwrite(final_img_des,resized_img)

    patch_h = 10240
    patch_w = 8192
    img_name = str(image_id)+'_seg_mask' 
    # extract_patches_from_binary_image_name3parts(img_from_patches,patch_h,patch_w, img_name, patch_saving_dir)


def create_heatmaps_from_roi_images_from_class_plus_seg_mask_fibrosis(roi_imagelogs_dir,patch_saving_dir):
    
    image_files = [x for x in sorted(os.listdir(roi_imagelogs_dir)) if x[-11:] == '_ac_img.jpg']
    
    #pdb.set_trace()
    name = image_files[0]        
    name_wo_ext_part1 = name.split('.')[0]
    actual_wsi_name  = name_wo_ext_part1.split('_')[0]
   

    num_rois = len(image_files)
    
    for k in range(num_rois):
        idv_roi_name = image_files[k]
        idv_name_wo_ext_part1 = idv_roi_name.split('.')[0]
        print('Processing for the ROI id of:',idv_name_wo_ext_part1)

        #idvactual_wsi_name  = idv_name_wo_ext_part1.split('_')[0]
        idv_roi_id = idv_name_wo_ext_part1.split('_')[1]
        #roi_id = int(name_wo_ext_part1.split('_')[-1])
        actual_roi_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_ac_mask.jpg'
        seg_roi_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_seg_mask_st.jpg'
        seg_mask_path = os.path.join(roi_imagelogs_dir,seg_roi_mask_name)
        roi_image = cv2.imread(roi_imagelogs_dir + idv_roi_name, cv2.IMREAD_UNCHANGED)
        roi_mask_input = cv2.imread(roi_imagelogs_dir + actual_roi_mask_name)
        roi_seg_mask_input = cv2.imread(roi_imagelogs_dir + seg_roi_mask_name)

        gray_mask_input = cv2.cvtColor(roi_mask_input, cv2.COLOR_BGR2GRAY)
        gray_seg_mask_input = cv2.cvtColor(roi_seg_mask_input, cv2.COLOR_BGR2GRAY)

        #roi_mask = 255.0*(gray_mask_input[:,:] >= 125)   
        #roi_mask = perform_erosion_operations(roi_mask)  
        roi_mask = 255.0*(gray_mask_input[:,:] >= 125)             
        roi_seg_mask = 255.0*(gray_seg_mask_input[:,:] >= 125)               

        # perform AND operation between seg-mask and actual-image mask
#        combined_seg_mask = cv2.bitwise_and(roi_mask, roi_seg_mask, mask = None) 
#        seg_roi_combined_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_combined_mask.jpg'
#        final_des_combined_mask = os.path.join(patch_saving_dir,seg_roi_combined_mask_name)
#        cv2.imwrite(final_des_combined_mask,combined_seg_mask)
#        
#        
#        #pdb.set_trace()
#        #pred_masks_seg = 255.0*(combined_seg_mask[:,:,:] >= 0.5)               
#        reconstructed_combined_seg_morph = perform_dilation_operations(combined_seg_mask) 
#        seg_roi_combined_with_morphology_mask_name = actual_wsi_name+'_'+str(idv_roi_id)+'_combined_wmorph_mask.jpg'
#        final_des_combined_mask_morph = os.path.join(patch_saving_dir,seg_roi_combined_with_morphology_mask_name)
#        cv2.imwrite(final_des_combined_mask_morph,reconstructed_combined_seg_morph)
        
        reconstructed_pred_seg_morph =cv2.imread(seg_mask_path) ## heatmap from the seg_class mask...
        #reconstructed_pred_seg_morph =cv2.imread(final_des_combined_mask_morph) ## heatmap from the combined mask...
        im_heatmap = create_heatmap(roi_image, reconstructed_pred_seg_morph, a1=.5, a2=.5)
        
        y_pred_name_seg_heatmap = actual_wsi_name+'_'+str(idv_roi_id)+'_heatmaps_st'+'.jpg' 
        final_des_pred_seg_heatmap = os.path.join(patch_saving_dir,y_pred_name_seg_heatmap)
        cv2.imwrite(final_des_pred_seg_heatmap,im_heatmap)
        
        #pdb.set_trace()
        total_pixels = roi_image.shape[0]*roi_image.shape[1]
        total_steatosis_pixels = np.sum(roi_seg_mask == 255)  
        total_number_tissue_pixels = np.sum(roi_mask == 255) 
        
        #pixels_wo_steatosis = total_number_tissue_pixels - total_steatosis_pixels
        percentage_of_steatosis_cells = (total_steatosis_pixels/total_number_tissue_pixels)*100
        
        if percentage_of_steatosis_cells == 'nan':
            percentage_of_steatosis_cells = 0
            
        print('The percentage of steatosis pixels :', percentage_of_steatosis_cells)
        ## Creat text file to save logs for the input regions....
        log_file_name_for_roi = actual_wsi_name+'_'+str(idv_roi_id)+'_logs_st.txt'
        roi_output_logs = join_path(patch_saving_dir,log_file_name_for_roi)        
        patch_logs = open(roi_output_logs, 'w')
         
        patch_logs.write("Total_pixels in ROI : "+str(total_pixels)
                            + "\n Total steatosis cells in ROI: " +str(total_steatosis_pixels)
                            + "\n Total tissue pixels in ROI:"+str(total_number_tissue_pixels)
                            + "\n Steatosis in percentage in ROI:"+str(percentage_of_steatosis_cells)
                            )

def convert_mask_to_svg(tissue, input_mask_file_path, closed_shape=False):

    if not tissue in ['steatosis', 'inflammation', 'ballooning', 'fibrosis']:
        tissue_class='noClass'
    elif tissue == 'steatosis':
        tissue_class = 'steatosis'
    elif tissue =='inflammation':
        tissue_class = 'inflammation'
    elif tissue =='ballooning':
        tissue_class = 'ballooning'
    elif tissue =='fibrosis':
        tissue_class = 'fibrosis'

    output_mask_file_path = os.path.splitext(input_mask_file_path)[0] + '.svg'

    # Read the image
    image = pyvips.Image.new_from_file(input_mask_file_path, access='sequential')
    imgray = np.ndarray(buffer=image.write_to_memory(),
                         dtype=np.uint8,
                         shape=[image.height, image.width, image.bands])
    # Find image size
    print(imgray.shape)
    height = imgray.shape[0]
    width = imgray.shape[1]

    # Find number of decimals saved in SVG file
    decimals_height = len(str(height)) - 2
    decimals_width = len(str(width)) - 2

    # Find number of patches (Divide the image into patches to avoid memory overflow)
    image_block_size = 64000
    patch_width = 1
    patch_height = 1
    if height * width > 2 * 1024**3:
        patch_height = math.ceil(height / image_block_size)
        patch_width = math.ceil(width / image_block_size)

    height_chunck = math.floor(height / patch_height)
    width_chunck = math.floor(width / patch_width)
    contours_list = []
    contours_list_opt = []
    origin_pos = np.zeros((1,1,2))
    for i in range (patch_height):
        for j in range (patch_width):
            f_i = i*height_chunck 
            f_j = j*width_chunck

            if i == patch_height:
                l_i = height - f_i
            else:
                l_i = (i+1)*height_chunck

            if j == patch_width:
                l_j = width - f_j
            else:
                l_j = (j+1)*width_chunck
            print(imgray.shape)
            img_patch = imgray[ f_i: l_i, f_j:l_j, :]
            if (imgray.shape[2] == 3):
                greyscale_img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
            else:
                greyscale_img_patch = img_patch[:, :, imgray.shape[2] - 1]
            
            ret, thresh = cv2.threshold(greyscale_img_patch,127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            new_contours = []
            opt_contours = []
            origin_pos[0, 0, 0] = f_j
            origin_pos[0, 0, 1] = f_i
            # eps = 0.50
            for k in contours:
                k[:, :, :] = k[:, :, :] + origin_pos

                # You can try more different parameters
                perimeter = cv2.arcLength(k,True)
                epsilon = 0.01*cv2.arcLength(k,True)
                approx = cv2.approxPolyDP(k,epsilon,True)

                opt_contours.append(approx)

                new_contours.append(k)

            contours_list.extend(new_contours)
            contours_list_opt.extend(opt_contours)
    print(f'old contour number: {len(contours_list)}')
    print(f'opt contour number: {len(contours_list_opt)}')
    f = open(output_mask_file_path, 'w+')
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" preserveAspectRatio="none" viewBox="0 0 100 100" width="100%" height="100%">')
    
    for elem in contours_list:
        if elem.shape[0] > 2: # Skip shapes with less than 3 points
            f.write(f'<path d="') # begin of a path
            for p in range(elem.shape[0]):
                x = round( 100 * elem[p, 0, 0] / width, decimals_height)
                y = round( 100 * elem[p, 0, 1] / height, decimals_width)
                if p == 0: # first node in a pathmust start with M
                    f.write(f'M{x} {y}')
                else: # the rest of points
                    f.write(f'L{x} {y}')
                    if  p == elem.shape[0] - 1:
                        if closed_shape: # verify whether it is the last point in a path
                            f.write(' z')
            f.write('"')
            f.write(f' class="{tissue_class}"') # end of points
            f.write('></path>') # end of a path
    f.write('</svg>')
    f.close()
    return output_mask_file_path

def zip(src_file_path):
    # Zip a file given by its absolute path with '.zip' extenssion
    zipfile_path = os.path.splitext(src_file_path)[0] + '.zip'
    z = zipfile.ZipFile(zipfile_path , 'w', zipfile.ZIP_DEFLATED)
    z.write(src_file_path, os.path.basename(src_file_path))
    z.close()
    return zipfile_path
