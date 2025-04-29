# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:11:34 2020

@author: deeplens
"""
import sys
sys.path.append('/opt/ASAP/bin')

import openslide
from scipy.misc import imsave, imresize
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np
import os
import os.path as osp
import json
import cv2
import multiresolutionimageinterface as mir
from scipy.misc import imsave
from PIL import Image, ImageDraw

from skimage.filters import threshold_otsu


import seaborn as sn
import shutil
import glob
from os.path import join as join_path
abspath = os.path.dirname(os.path.abspath(__file__))

save = True

valid_images = ['.svs','.tif','.jpg','.png']

import pdb


def extract_HPFs_mask(full_img,full_mask,patch_h,patch_w, img_name, img_saving_dir):
    
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
            f_img_name =str(img_name)+'_'+str(pn)+'.png'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.png'           
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

def generate_mask_for_WSI(tumor_paths,anno_mask_paths,mask_saving_dir,wsi_image_name):
    
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(tumor_paths)
    annotation_list=mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(anno_mask_paths)
    xml_repository.load()
    annotation_mask=mir.AnnotationToMask()
    camelyon17_type_mask = False
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 255, '_1': 255, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
    output_path= osp.join(mask_saving_dir, osp.basename(wsi_image_name.replace('.tif', '_mask.tif')))
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)
    mask_final_path = output_path
    
    return mask_final_path

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

def extact_patches_mask_from_tumor():
    
    #slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/tumor/tumors_051_111/rem'
    slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/tumor/tumors_001_050_wsi/only_two/'
    mask_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/masks/44_50_masks'
    slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
    slide_paths.sort()
    mask_paths = glob.glob(osp.join(mask_path, '*.tif'))
    mask_paths.sort()
    
    img_saving_dir = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/HPFs_images_masks_camelyon16/tumor/HPFs_51_110/'
    
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
        extract_HPFs_mask(rgb_imagenew,mask_image,patch_h,patch_w, img_name, img_saving_dir)
        scan_id = scan_id + 1


def extact_patches_from_normal():    
    slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/training_samples_and_annotation/normal/example_wsi/'
    slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
    slide_paths.sort()
    img_saving_dir = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/HPFs_images_masks_camelyon16/normal/'
    patch_h =1024
    patch_w = 1024
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
        extract_HPFs_from_normal(rgb_imagenew,patch_h,patch_w, img_name, img_saving_dir)
        scan_id = scan_id + 1


def get_normal_image_contours(cont_img, rgb_image, cont_img_tmp):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_tmp, _ = cv2.findContours(cont_img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # print(boundingBoxes)
    contours_rgb_image_array = np.array(rgb_image)
    contours_rgb_image_array_tmp = np.array(rgb_image)

    line_color = (255, 0, 0)  # blue color code
    cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
    cv2.drawContours(contours_rgb_image_array_tmp, contours_tmp, -1, line_color, 3)
    # cv2.drawContours(mask_image, contours_mask, -1, line_color, 3)
    return contours_rgb_image_array, boundingBoxes, contours_rgb_image_array_tmp

#def extract_random_patches_from_dounding_box_for_normal_tissu(rgb_image,bounding_boxes,patch_h,patch_w,patch_saving_path):
#    
#    
#    #mag_factor = pow(2, self.level_used)
#    # extracting patches from ...rgb imagess...
#    for i, bounding_box in enumerate(bounding_boxes):
#        
#        if int(bounding_box[2])>patch_w and int(bounding_box[3])>patch_h):
#            b_x_start = int(bounding_box[0])
#            b_y_start = int(bounding_box[1]) 
#            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) #* mag_factor
#            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) #* mag_factor
#            
#            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
#            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
#            # X = np.arange(b_x_start, b_x_end-256, 5)
#            # Y = np.arange(b_y_start, b_y_end-256, 5)
#            for x, y in zip(X, Y):
#                patch = self.wsi_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
#                mask = self.mask_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
#                mask_gt = np.array(mask)
#                # mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
#                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
#                patch_array = np.array(patch)
#                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
#    
#                    if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
#                        patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
#                        lower_red = np.array([20, 20, 20])
#                        upper_red = np.array([200, 200, 200])
#                        mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
#                        white_pixel_cnt = cv2.countNonZero(mask_patch)
#    
#                        if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
#                            # mask = Image.fromarray(mask)
#                            patch.save(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + PATCH_NORMAL_PREFIX +
#                                       str(self.negative_patch_index), 'PNG')
#                            # mask.save(PROCESSED_PATCHES_NORMAL_PATH + PATCH_NORMAL_PREFIX + str(self.patch_index),
#                            #           'PNG')
#                            self.negative_patch_index += 1
#                    else:  # mask_gt contains tumor area
#                        if white_pixel_cnt_gt >= ((PATCH_SIZE * PATCH_SIZE) * 0.85):
#                            patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_TUMOR_PREFIX +
#                                       str(self.positive_patch_index), 'PNG')
#                            self.positive_patch_index += 1

def extract_HPFs_from_roi_normal_tissue(full_img, patch_h, patch_w, img_name, img_saving_dir):
    
#    if not os.path.isdir("%s/%s"%(img_saving_dir,img_name)):
#        os.makedirs("%s/%s"%(img_saving_dir,img_name))            
#    patches_saving_dir = join_path(img_saving_dir+img_name+'/')
#    
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)    
    #pdb.set_trace()
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]            
            patch_hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])
            mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
            white_pixel_cnt = cv2.countNonZero(mask_patch)
           
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            final_des_img = os.path.join(img_saving_dir,f_img_name)
            
            if white_pixel_cnt > ((patch_h*patch_w) * 0.50):
                cv2.imwrite(final_des_img,patch_img)
                pn+=1 
            #cv2.imwrite(final_des_img,patch_img)
            #pn+=1
            
            #roi_log = {}
#            roi_log["ID"] = image_name
#            roi_log["roi_ID"] = box_id
#            roi_log["roi_height"] = roi_height
#            roi_log["roi_width"] = roi_width
#            roi_log["patch_width"] = patch_w
#            roi_log["patch_height"] = patch_h
#            roi_log["roi_no_patches_x_axis"] = num_patch_per_row
#            roi_log["roi_no_patches_y_axis"] = num_patch_per_column
#            roi_log["roi_starting_column"] = b_y_start
#            roi_log["roi_starting_row"] = b_x_start
#
#            # make experimental log saving path...
#            json_file = os.path.join(other_logs_saving_path,f_image_name+'_roi_patching_log.json')
#            with open(json_file, 'w') as file_path:
#                json.dump(str(roi_log), file_path, indent=4, sort_keys=True)
            

      
        k +=1
        print ('Processing for: ' +str(k))

    return pn

def extract_HPFs_from_roi_normal_tissue_testing_phase(full_img, patch_h, patch_w):
 
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)    
    
    patches_per_roi = []  
    k = 0
    pn = 0
    
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]        
            patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGBA2BGR)
            #patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
            patch_hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])
            mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
            white_pixel_cnt = cv2.countNonZero(mask_patch)           
         
            #pdb.set_trace()           
            if white_pixel_cnt > ((patch_h*patch_w) * 0.50):
                patches_per_roi.append(patch_img)
                #patches_per_roi[pn] =patch_img 
                pn += 1
        k +=1
        print ('Processing for: ' +str(k))

    return patches_per_roi
    
    
def extract_HPFs_from_roi_regions(full_img, box_id, roi_x_start, roi_y_start, roi_height, roi_width, patch_h, patch_w, img_name, img_saving_dir,other_logs_saving_path):
     
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)    
    #pdb.set_trace()
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]            
            patch_hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([20, 20, 20])
            upper_red = np.array([200, 200, 200])
            mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
            white_pixel_cnt = cv2.countNonZero(mask_patch)     
            
            f_img_name_wo_ext =str(img_name)+'_'+str(pn)
            f_img_name =f_img_name_wo_ext+'.jpg'
            final_des_img = os.path.join(img_saving_dir,f_img_name)
            
            p_x_start = roi_x_start + r_s*patch_h
            p_y_start = roi_y_start + c_s*patch_w
            
            patch_log = {}
            
      
            
            if white_pixel_cnt > ((patch_h*patch_w) * 0.50):
                cv2.imwrite(final_des_img,patch_img)
                pn+=1             
               
                #pdb.set_trace()
                
                patch_log["ID"] = f_img_name_wo_ext
                patch_log["roi_ID"] = box_id
                patch_log["roi_height"] = roi_height
                patch_log["roi_width"] = roi_width
                patch_log["patch_width"] = patch_w
                patch_log["patch_height"] = patch_h
                patch_log["roi_no_patches_x_axis"] = rows
                patch_log["roi_no_patches_y_axis"] = columns
                patch_log["roi_starting_y"] = roi_y_start
                patch_log["roi_starting_x"] = roi_x_start         
                patch_log["patch_starting_y"] = p_y_start
                patch_log["patch_starting_x"] = p_x_start
                # make experimental log saving path...
                json_file = os.path.join(other_logs_saving_path,f_img_name_wo_ext+'_logs.json')
                with open(json_file, 'w') as file_path:
                    json.dump(str(patch_log), file_path, indent=4, sort_keys=True)      
        k +=1
        print ('Processing for: ' +str(k))

    return pn
    

                       
def extract_patches_from_ROI(rgb_image,bounding_boxes,patch_h,patch_w,image_name,patch_saving_path,other_logs_saving_path):
        
    # extracting patches from ...rgb imagess...    
        
    box_id = 0
    for i, bounding_box in enumerate(bounding_boxes):
        
        if (int(bounding_box[2])>patch_w and int(bounding_box[3])>patch_h):
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1]) 
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2]))
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3]))
            
            roi_height = int(bounding_box[2])
            roi_width = int(bounding_box[3])
            
            num_patch_per_column = int(b_x_end/patch_w)
            num_patch_per_row = int(b_y_end/patch_h)
            f_x_end = (int(bounding_box[0]) + int(num_patch_per_column*patch_w))
            f_y_end = (int(bounding_box[1]) + int(num_patch_per_row*patch_h))

            #roi_rgb_image = rgb_image
            #roi_rgb_image = rgb_image[b_x_start: f_x_end,b_y_start:f_y_end,:]
            roi_rgb_image = rgb_image[b_y_start:f_y_end,b_x_start: f_x_end,:]

            f_image_name = image_name+'_'+str(box_id)
            #pdb.set_trace()
            #extract_HPFs_from_roi_regions(roi_rgb_image, patch_h, patch_w, f_image_name, patch_saving_path)
            extract_HPFs_from_roi_regions(roi_rgb_image, box_id, b_x_start, b_y_start, roi_height, roi_width, patch_h, patch_w, f_image_name, patch_saving_path,other_logs_saving_path)

                
        else:
            #mid_point_x = int(bounding_box[2]/)
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1]) 
            f_x_end = (int(bounding_box[0]) + int(patch_w))
            f_y_end = (int(bounding_box[1]) + int(patch_h))     
            
            roi_height = int(bounding_box[2])
            roi_width = int(bounding_box[3])
            #roi_rgb_image = rgb_image[b_x_start: f_x_end,b_y_start:f_y_end,:]
            roi_rgb_image = rgb_image[b_y_start:f_y_end,b_x_start: f_x_end,:]
            #pdb.set_trace()
            f_image_name = image_name+'_'+str(box_id)
            #extract_HPFs_from_roi_regions(roi_rgb_image, patch_h, patch_w, f_image_name, patch_saving_path)
            extract_HPFs_from_roi_regions(roi_rgb_image, box_id, b_x_start, b_y_start, roi_height, roi_width, patch_h, patch_w, f_image_name, patch_saving_path,other_logs_saving_path)
      

        box_id = box_id+1
            
def extract_HPFs_from_dounding_box_for_normal_tissu_testing_phase(rgb_image,bounding_boxes,patch_h,patch_w):
    
    # extracting patches from ...rgb imagess...
    pathes_from_wsi = []     
    box_id = 0
    for i, bounding_box in enumerate(bounding_boxes):
        
        if (int(bounding_box[2])>patch_w and int(bounding_box[3])>patch_h):
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1]) 
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2]))
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3]))
            
            num_patch_per_column = int(b_x_end/patch_w)
            num_patch_per_row = int(b_y_end/patch_h)
            f_x_end = (int(bounding_box[0]) + int(num_patch_per_column*patch_w))
            f_y_end = (int(bounding_box[1]) + int(num_patch_per_row*patch_h))

            roi_rgb_image = rgb_image[b_y_start:f_y_end,b_x_start: f_x_end,:]
            
            #pdb.set_trace()
            #f_image_name = image_name+'_'+str(box_id)
            patche_per_roi = extract_HPFs_from_roi_normal_tissue_testing_phase(roi_rgb_image, patch_h, patch_w)
        else:
            #mid_point_x = int(bounding_box[2]/)
            b_x_start = int(bounding_box[0])
            b_y_start = int(bounding_box[1]) 
            f_x_end = (int(bounding_box[0]) + int(patch_w))
            f_y_end = (int(bounding_box[1]) + int(patch_h))            
            #roi_rgb_image = rgb_image[b_x_start: f_x_end,b_y_start:f_y_end,:]
            roi_rgb_image = rgb_image[b_y_start:f_y_end,b_x_start: f_x_end,:]
            #pdb.set_trace()
            #f_image_name = image_name+'_'+str(box_id)
            patche_per_roi = extract_HPFs_from_roi_normal_tissue_testing_phase(roi_rgb_image, patch_h, patch_w)
        
        pathes_from_wsi.append(patche_per_roi) 
        #pathes_from_wsi[wsi_patch_id:wsi_patch_id+num_patches_roi] = patche_per_roi
        box_id = box_id+1
    
    
    return pathes_from_wsi
               
                                      
def extract_roi_from_normal_wsi():
    
    #thresh = threshold_otsu(thumbnail_grey)
    #binary = thumbnail_grey > thresh
           
    slide_path = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/training_samples_and_annotation/normal/example_wsi/'
    slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
    slide_paths.sort()
    img_saving_dir = '/home/deeplens/deeplens_projects/metastatis_detection/database/CAMELYON16/samples_annotation/HPFs_images_masks_camelyon16/normal/normal_patches/'
    
    patch_h =128
    patch_w = 128
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
        rgb_image = slide.read_region((0, 0), lavel, slide.level_dimensions[lavel])
        rgb_imagenew = np.array(rgb_image)
        # extract foreground mask
        
        hsv = cv2.cvtColor(rgb_imagenew, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([30, 30, 30])
        # [255, 255, 255]
        upper_red = np.array([200, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(rgb_imagenew, rgb_imagenew, mask=mask)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        close_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        image_close_tmp = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel_tmp))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        open_kernel_tmp = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        image_open_tmp = Image.fromarray(cv2.morphologyEx(np.array(image_close_tmp), cv2.MORPH_OPEN, open_kernel_tmp))
        

        
        contour_rgb, bounding_boxes, contour_rgb_tmp = get_normal_image_contours(np.array(image_open), rgb_imagenew, np.array(image_open_tmp))
          
        #pdb.set_trace()                                                                            
        extract_HPFs_from_dounding_box_for_normal_tissu(rgb_imagenew,bounding_boxes,patch_h,patch_w,img_name,img_saving_dir)
        
        print('Processing for image id: ' + str(img_name))
        
        #extract_HPFs_mask(rgb_imagenew,foreground_binary,patch_h,patch_w, img_name, img_saving_dir)
        scan_id = scan_id + 1
        
    
def save_selected_normal_patches_masks_subdir_from_seg_model(patches_from_wsi,pred_masks_seg,slide_name,patch_saving_path):
    
    if not os.path.isdir("%s/%s"%(patch_saving_path,'patches'+'/')):
       os.makedirs("%s/%s"%(patch_saving_path,'patches'+'/'))   
    selected_normal_patch_saving_path_final = join_path(patch_saving_path,'patches'+'/') 
    
    if not os.path.isdir("%s/%s"%(patch_saving_path,'masks'+'/')):
       os.makedirs("%s/%s"%(patch_saving_path,'masks'+'/'))   
    selected_normal_mask_saving_path_final = join_path(patch_saving_path,'masks'+'/') 
       
    num_samples,patch_h, patch_w,c = patches_from_wsi.shape 
    #pdb.set_trace()
    #num_samples = len(patches_from_wsi)
    patch_id = 0   
    for image_id in range(num_samples):        
        patch_img = patches_from_wsi[image_id,:,:,:]
        patch_mask = pred_masks_seg[image_id,:,:]
        white_pixel_cnt = cv2.countNonZero(patch_mask)           
        f_img_name =str(slide_name)+'_'+str(patch_id)+'.png'
        f_mask_name =str(slide_name)+'_'+str(patch_id)+'_mask.png'
        final_des_img = os.path.join(selected_normal_patch_saving_path_final,f_img_name)
        final_des_mask = os.path.join(selected_normal_mask_saving_path_final,f_mask_name)

        if white_pixel_cnt > 10:
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
            patch_id+=1 
    
    image_path = selected_normal_patch_saving_path_final
    mask_path = selected_normal_mask_saving_path_final
    
    return image_path,mask_path

def save_selected_pred_masks_subdir_from_seg_model(patches_from_wsi,pred_masks_seg,patches_name,patch_saving_path,Logs_sources):
    
    if not os.path.isdir("%s/%s"%(patch_saving_path,'patches'+'/')):
       os.makedirs("%s/%s"%(patch_saving_path,'patches'+'/'))   
    selected_normal_patch_saving_path_final = join_path(patch_saving_path,'patches'+'/') 
    
    if not os.path.isdir("%s/%s"%(patch_saving_path,'masks'+'/')):
       os.makedirs("%s/%s"%(patch_saving_path,'masks'+'/'))   
    selected_normal_mask_saving_path_final = join_path(patch_saving_path,'masks'+'/') 
    
    if not os.path.isdir("%s/%s"%(patch_saving_path,'logs'+'/')):
       os.makedirs("%s/%s"%(patch_saving_path,'logs'+'/'))   
    selected_normal_logs_saving_path_final = join_path(patch_saving_path,'logs'+'/') 
    
       
    num_samples,patch_h, patch_w,c = patches_from_wsi.shape 
    total_num_pixels = patch_h*patch_w
    #pdb.set_trace()
    #num_samples = len(patches_from_wsi)
    patch_id = 0   
    for image_id in range(num_samples):        
        patch_img = patches_from_wsi[image_id,:,:,:]
        patch_mask = pred_masks_seg[image_id,:,:]
        white_pixel_cnt = cv2.countNonZero(patch_mask) 

        patch_name = patches_name[image_id] 
         #f_img_name =str(slide_name)+'_'+str(patch_id)+'.png'
        f_img_name = patch_name
        img_name_wo_ext = f_img_name.split('.')[0]        
          # Find the json log fle for patches and move to the desire directory...
        json_file_name = img_name_wo_ext+'_logs.json'
        json_file_name_with_dir = join_path(Logs_sources,json_file_name) 
       
        f_mask_name =img_name_wo_ext+'_mask.png'
        final_des_img = os.path.join(selected_normal_patch_saving_path_final,f_img_name)
        final_des_mask = os.path.join(selected_normal_mask_saving_path_final,f_mask_name)

        percentage_of_pixels = total_num_pixels*0.1
        
        if white_pixel_cnt > percentage_of_pixels:
            
            print('Number of white pixels : '+str(white_pixel_cnt))
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)            
            # move the json file as well
            shutil.move(json_file_name_with_dir,selected_normal_logs_saving_path_final)
            patch_id+=1 
    
    image_path = selected_normal_patch_saving_path_final
    mask_path = selected_normal_mask_saving_path_final
    logs_path = selected_normal_logs_saving_path_final
    
    return image_path,mask_path,logs_path
    
def save_selected_normal_patches_masks_from_seg_model(patches_from_wsi,pred_masks_seg,slide_name,patch_saving_path):
           
    num_samples,patch_h, patch_w,c = patches_from_wsi.shape 
    #pdb.set_trace()
    #num_samples = len(patches_from_wsi)
    patch_id = 0   
    for image_id in range(num_samples):        
        patch_img = patches_from_wsi[image_id,:,:,:]
        patch_mask = pred_masks_seg[image_id,:,:]
        white_pixel_cnt = cv2.countNonZero(patch_mask)           
        f_img_name =str(slide_name)+'_'+str(patch_id)+'.png'
        f_mask_name =str(slide_name)+'_'+str(patch_id)+'_mask.png'
        final_des_img = os.path.join(patch_saving_path,f_img_name)
        final_des_mask = os.path.join(patch_saving_path,f_mask_name)

        if white_pixel_cnt > (patch_h*patch_w)*0.25:
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
            patch_id+=1     
    return 0
    
def save_selected_normal_patches_from_class_model(x_test_class,y_hat_class,thresh_value,slide_name,patch_saving_path):
           
    num_samples,patch_h, patch_w,c = x_test_class.shape 
    
    #pdb.set_trace()
    #num_samples = len(patches_from_wsi)
    patch_id = 0   
    for image_id in range(num_samples):        
        patch_img = x_test_class[image_id,:,:,:]
        #patch_mask = pred_masks_seg[image_id,:,:]
        #white_pixel_cnt = cv2.countNonZero(patch_mask)           
        f_img_name =str(slide_name)+'_'+str(patch_id)+'_hnp'+'.png'
        #f_mask_name =str(slide_name)+'_'+str(patch_id)+'_mask.png'
        final_des_img = os.path.join(patch_saving_path,f_img_name)
        #final_des_mask = os.path.join(patch_saving_path,f_mask_name)
        conf_arr = y_hat_class[image_id,:]
        conf_arr = np.array(conf_arr)
        conf_value = np.amax(conf_arr)           
        conf_index = np.where(conf_arr == np.amax(conf_arr))
        conf_index_final = int(conf_index[0])
        if conf_value > thresh_value and conf_index_final == 1:
            print('Condition satisfied for the values : '+str(conf_value)+' and class : '+str(conf_index_final))
            cv2.imwrite(final_des_img,patch_img)
            patch_id+=1 

    return patch_saving_path



def save_selected_patches_from_class_model(x_test_class,patches_name,y_hat_class,thresh_value,Logs_sources,patch_saving_path,LSP):
           
    num_samples,patch_h, patch_w,c = x_test_class.shape 
      
    #pdb.set_trace()
    #num_samples = len(patches_from_wsi)
    patch_id = 0   
    for image_id in range(num_samples-1):        
        patch_img = x_test_class[image_id,:,:,:]
        f_img_name = patches_name[image_id]
        f_img_name_wo_ext = f_img_name.split('.')[0]
        
        # Find the json log fle for patches and move to the desire directory...
        json_file_name = f_img_name_wo_ext+'_logs.json'
        json_file_name_with_dir = join_path(Logs_sources,json_file_name) 

        final_des_img = os.path.join(patch_saving_path,f_img_name)
        
        conf_arr = y_hat_class[image_id,:]
        conf_arr = np.array(conf_arr)
        conf_value = np.amax(conf_arr)           
        conf_index = np.where(conf_arr == np.amax(conf_arr))
        conf_index_final = int(conf_index[0])
        if conf_value > thresh_value and conf_index_final == 1:
            print('Condition satisfied for the values : '+str(conf_value)+' and class : '+str(conf_index_final))
            cv2.imwrite(final_des_img,patch_img)           
            # move the json file as well
            shutil.move(json_file_name_with_dir,LSP)
            patch_id+=1 
            
    num_patches = patch_id
    
    return patch_saving_path,LSP,num_patches


    
def read_image_from_dir(image_dir,img_h,img_w):

    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.jpg' or '.png']
    total = len(all_images)
    ac_imgs = np.ndarray((total, img_h,img_w,3), dtype=np.uint8)

    k = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for i, image_name in enumerate(all_images):
         ac_img = cv2.imread(os.path.join(image_dir, image_name))                
         ac_imgs[k] = ac_img 
         k += 1
         print ('Reading done',i)
     
    """
    perm = np.random.permutation(len(imgs_mask))
    imgs = imgs[perm]
    imgs_mask = imgs_mask[perm]
    ac_imgs = ac_imgs[perm]
    """
    image_names = all_images

    return ac_imgs,image_names
    
def delete_image_files_from_dir(directory):  
    
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if file.endswith('.jpg' or '.png')]   
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


def delete_log_files_from_dir(directory):    
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if file.endswith('.json')]   
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)
        
def select_index_for_selected_patches(y_pred_model):
    
    total_sampels = len(y_pred_model)
    index_array = []
    for k in range(total_sampels):
        conf_arr_idv = np.array(y_pred_model[k,:])
        conf_value = np.amax(conf_arr_idv)                  
        conf_index = np.where(conf_arr == np.amax(conf_arr))
        conf_index_final = int(conf_index[0])
        if conf_value > thresh_value and conf_index_final == 0:
            index_array.append(k)
            
    selected_indexes = np.array(index_array)
    
    return selected_indexes

#def generate_mask_for_wsi():
def generate_predicted_mask_for_wsi(classified_and_seg_masks_dir,classified_and_seg_log_dir,mask_height,mask_width,patch_h,patch_w,slide_name,mask_final_path):
    
    pred_wsi_mask = np.ndarray((mask_height,mask_width), dtype=np.uint8)   
    files_in_directory = os.listdir(classified_and_seg_log_dir)
    filtered_files = [file for file in files_in_directory if file.endswith('.json')]  
    
    for file in filtered_files:
        
        mask_name = file.split('_logs')[0]
        f_mask_name = mask_name+'_mask.png'       
        mask_img = cv2.imread(os.path.join(classified_and_seg_masks_dir,f_mask_name))                
       

        fields_img = mask_img.shape
        print('Image Id : ', mask_name)
        
        if len(fields_img)>2:
            gary_mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            final_pred_mask = 255.0*(gary_mask_img[:,:] >= gary_mask_img.mean()) 
        else:
            mx_val = mask_img.max() 
            if mx_val > 10:
                final_pred_mask = mask_img 
            else:
                final_pred_mask = 255.0*(mask_img[:,:] >= 0.5) 

        json_path_to_file = os.path.join(classified_and_seg_log_dir, file)
        
        with open(json_path_to_file, "r") as f_json:
            patch_logs = json.load(f_json)
            
        splited_patch_logs = patch_logs.split(',')
        x_val = splited_patch_logs[10].split(':')[1]
        y_val_with_pth = splited_patch_logs[11].split(':')[1]
        y_val_wo_pth = y_val_with_pth.split('}')[0]
        y_position = int(x_val)
        x_position = int(y_val_wo_pth)
        
        print('x_value = ', x_position)
        print('y_value = ', y_position)


        pred_wsi_mask[y_position:y_position+patch_h,x_position:x_position+patch_w] = final_pred_mask
    
    #hight_new = int(mask_height/4)
    #width_new = int(mask_width/4)
    
    
    hight_new = 10240
    width_new = 8192
    
    pred_mask_4saving = cv2.resize(pred_wsi_mask, (hight_new,width_new), interpolation = cv2.INTER_AREA)
    
    pred_mask_saving_path = os.path.jion(mask_final_path,str(slide_name)+'_pred_wsi_mask.jpg')
    
    cv2.imwrite(pred_mask_saving_path,pred_mask_4saving)

    return pred_wsi_mask


def compared_wsi_masks_calculate_number_FPP(gt_wsi_mask, pred_wsi_mask, patch_h, patch_w):
    
    gt_wsi_mask = 255.0*(gt_wsi_mask[:,:] >= gt_wsi_mask.mean()) 
    pred_wsi_mask = 255.0*(pred_wsi_mask[:,:] >= pred_wsi_mask.mean()) 

    slide_height,slide_width = gt_wsi_mask.shape
    
    FPDC = 0
    for y in range(0,slide_height,patch_h):
        for x in range(0,slide_width,patch_w):          
            gt_patch = gt_wsi_mask[y:y+patch_h,x:x+patch_w]
            pred_patch = pred_wsi_mask[y:y+patch_h,x:x+patch_w]
            
            gt_mx_val = gt_patch.max() 
            pred_mx_val = pred_patch.max()
            
            if pred_mx_val > 0 and gt_mx_val ==0:
                FPDC = FPDC + 1
    
    return FPDC
                
        
#def main():
#    
#    #extact_patches_mask_from_tumor()
#    #extact_patches_from_normal()
#    extract_roi_from_normal_wsi()
#    
#    #generate_predicted_mask_for_wsi(classified_and_seg_masks_dir,classified_and_seg_log_dir,mask_height,mask_width,patch_h,patch_w)
#
#    
#
#if __name__== "__main__": 
#    # call the main function..
#    main()

