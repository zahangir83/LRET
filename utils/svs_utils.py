# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:54:37 2019
@author: deeplens
"""
import openslide
from scipy.misc import imsave, imresize
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np
import os
import pdb
import json
#import cv2
from scipy.misc import imsave
from os.path import join as join_path
abspath = os.path.dirname(os.path.abspath(__file__))

save = True

valid_images = ['.svs','.jpg']


def extract_patches_from_image(full_img,patch_h,patch_w, img_name, patches_saving_dir):

    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)       
    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            f_img_name =str(pn)+'.jpg'
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



def extract_ss_patches_from_svs(svs_img_dir, patches_saving_dir, patch_size):
            
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
        
    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
    
    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    
    image_svs = [x for x in sorted(os.listdir(svs_img_dir)) if x[-4:] == '.svs' or '.tif']
        
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
                print("Processing HPFs  # "+str(img_saving_idx))                
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
    svs_log["id_rows_columns"] = starting_row_columns
     
    # make experimental log saving path...
    json_file = os.path.join(patches_sub_dir,'svs_patching_log.json')
    with open(json_file, 'w') as file_path:
        json.dump(svs_log, file_path, indent=4, sort_keys=True)
        
    images_list =  starting_row_columns
    
    return patches_sub_dir,images_list

def extract_all_patches_from_svs(svs_img_dir, patches_saving_dir, patch_size):
            
    patch_dir_name = 'patches_'+str(patch_size[0])+'/'
        
    if not os.path.isdir("%s/%s"%(patches_saving_dir,patch_dir_name)):
        os.makedirs("%s/%s"%(patches_saving_dir,patch_dir_name))        
    
    patches_dir = join_path(patches_saving_dir+patch_dir_name)
    
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
    imsave(final_img_des,img_from_patches)
    
    
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