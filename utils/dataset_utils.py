#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:57:07 2018
@author: zahangir
"""
import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import cv2
from os.path import join as join_path
import pdb
from collections import defaultdict
#from skimage.transform import resize
#import shutil
import scipy.ndimage as ndimage
import tensorflow as tf

kernel = np.ones((6,6), np.uint8) 

allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.mat','*.tif']

def color_perturbation(image):
      
      image = tf.image.random_brightness(image, max_delta=64./ 255.)
      image = tf.image.random_saturation(image, lower=0, upper=0.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, lower=0, upper=0.75)      
      
      return image
      
def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x

def samples_normalization (x_data, y_data):
    x_data = x_data.astype('float32')
    mean = np.mean(x_data)  # mean for data centering
    std = np.std(x_data)  # std for data normalization
    x_data -= mean
    x_data /= std
    
    y_data = y_data.astype('float32')
    y_data /= 255.  # scale masks to [0, 1]
    return x_data,y_data,mean,std
    

def split_data_train_val (ac_x_data,x_data,y_data):

    sample_count = len(x_data)   
    train_size = int(sample_count * 4.8 // 5)    
    
    ac_x_train = ac_x_data[:train_size]
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    
    ac_x_val = ac_x_data[train_size:]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    
    return ac_x_train,x_train,y_train,ac_x_val,x_val,y_val


def applyImageAugmentationAndRetrieveGenerator():


    # We create two instances with the same arguments
    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2
                         )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory('dataset/train_images',
                                                        target_size=(360,480),    
                                                        class_mode=None,
                                                        seed=seed,
                                                        batch_size = 32)
    
    mask_generator = mask_datagen.flow_from_directory('dataset/train_masks',
                                                      target_size=(360,480),  
                                                      class_mode=None,
                                                      seed=seed,
                                                      batch_size = 32)
    

    train_generator = zip(image_generator, mask_generator)
    
    return train_generator

def extract_image_patches(full_img,full_mask,patch_h,patch_w, img_name, imd_saving_dir):
        
    height,width, channel = full_img.shape    
    rows = (int) (height/patch_h)
    columns = (int) (width/patch_w)

    k = 0
    pn = 0
    for r_s in range(rows):
        for c_s in range(columns):
            patch_img = full_img[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w,:]
            patch_mask = full_mask[r_s*patch_h: r_s*patch_h+patch_h,c_s*patch_w:c_s*patch_w+patch_w]
            
            patch_img_flip_lr = np.fliplr(patch_img)         
            patch_mask_flip_lr = np.fliplr(patch_mask)
            
            patch_img_flip_up = np.flipud(patch_img)
            patch_mask_flip_up = np.flipud(patch_mask)
            
            f_img_name =str(img_name)+'_'+str(pn)+'.jpg'
            f_mask_name =str(img_name)+'_'+str(pn)+'_mask'+'.jpg'           
            final_des_img = os.path.join(imd_saving_dir,f_img_name)
            final_des_mask = os.path.join(imd_saving_dir,f_mask_name)
            
            cv2.imwrite(final_des_img,patch_img)
            cv2.imwrite(final_des_mask,patch_mask)
            
            f_img_name_lr =str(img_name)+'_'+str(pn)+'_lr.jpg'
            f_mask_name_lr =str(img_name)+'_'+str(pn)+'_lr_mask'+'.jpg'           
            final_des_img_lr = os.path.join(imd_saving_dir,f_img_name_lr)
            final_des_mask_lr = os.path.join(imd_saving_dir,f_mask_name_lr)
            
            
            cv2.imwrite(final_des_img_lr,patch_img_flip_lr)
            cv2.imwrite(final_des_mask_lr,patch_mask_flip_lr)
            
            
            f_img_name_up =str(img_name)+'_'+str(pn)+'_up.jpg'
            f_mask_name_up =str(img_name)+'_'+str(pn)+'_up_mask'+'.jpg'           
            final_des_img_up = os.path.join(imd_saving_dir,f_img_name_up)
            final_des_mask_up = os.path.join(imd_saving_dir,f_mask_name_up)
            
              
            cv2.imwrite(final_des_img_up,patch_img_flip_up)
            cv2.imwrite(final_des_mask_up,patch_mask_flip_up)
            
            #mx_val = patch_mask.max()
            #mn_val = patch_mask.min()
            #print ('max_val : '+str(mx_val))
            #print ('min_val : '+str(mn_val))          

            #if mx_val > 10:

            pn+=1
            
        k +=1
        print ('Processing for: ' +str(k))

    return pn

def read_single_pixel_anno_data(image_dir,img_h,img_w):

    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.jpg']
    
    total = int(np.round(len(all_images)/2))

    ac_imgs = np.ndarray((total, img_h,img_w,3), dtype=np.uint8)
    imgs = np.ndarray((total, img_h,img_w), dtype=np.uint8)
    imgs_mask = np.ndarray((total,img_h,img_w), dtype=np.uint8)
    k = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for i, image_name in enumerate(all_images):
         if 'mask' in image_name:
             continue
         image_mask_name = image_name.split('.')[0] + '_mask.jpg'
          # patient_num = image_name.split('_')[0]
         img = cv2.imread(os.path.join(image_dir, image_name), cv2.IMREAD_GRAYSCALE)
         ac_img = cv2.imread(os.path.join(image_dir, image_name))
         img_mask = cv2.imread(os.path.join(image_dir, image_mask_name), cv2.IMREAD_GRAYSCALE)
         img_mask = 255.0*(img_mask[:,:]> 0)
         img_mask = cv2.dilate(img_mask,kernel,iterations = 1)
         img_mask = ndimage.gaussian_filter(img_mask, sigma=(1,1),order = 0)     
         img_mask = 255.0*(img_mask[:,:]> 0)
        
         
         ac_imgs[k] = ac_img 
         imgs[k] = img
         imgs_mask[k] = img_mask

         k += 1
         print ('Done',i)
     
    """
    perm = np.random.permutation(len(imgs_mask))
    imgs = imgs[perm]
    imgs_mask = imgs_mask[perm]
    ac_imgs = ac_imgs[perm]
    """
    return ac_imgs, imgs, imgs_mask

def create_dataset_patches_driver(image_dir,saving_dir,patch_h,patch_w):
        
    all_images = [x for x in sorted(os.listdir(image_dir)) if x[-4:] == '.jpg']
    

    pdb.set_trace()
     
    Total_patches = 0

    for i, name in enumerate(all_images):
        
        if 'morh_banary' in name:
            continue
          
        im = cv2.imread(image_dir + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    
        #im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
        acc_name = name.split('.')[0]
        mask_name = acc_name +'_morh_banary.jpg'       
        mask_im = cv2.imread(image_dir + mask_name, cv2.IMREAD_UNCHANGED) #.astype('float32')/255.
        #mask_im = cv2.resize(mask_im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        #y_data[i] = mask_im
        
        mask_im = 255*(mask_im[:,:]> 20)
        img_rz = im
        img_mask_rz = mask_im
        
       
        
        num_patches = extract_image_patches (img_rz, img_mask_rz, patch_h, patch_w, acc_name, saving_dir)
        
        print ('Processing for: ' +str(i))
        Total_patches = Total_patches + num_patches
    
    return 0


def read_testing_images(data_path,image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    #images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))
    images = glob.glob(train_data_path + "/*.jpg")
    total = np.round(len(images)) 

    acc_imgs = np.ndarray((total, image_h, image_w,3), dtype=np.uint8)
    gray_mgs = np.zeros((total, image_h, image_w), dtype=np.uint8)
    #imgs_mask = np.zeros((total, image_h, image_w), dtype=np.uint8)
    
    i = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:
        '''
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.jpg'
        '''
        #image_mask_name = image_name.split('/')[-1]      
        #img_first = image_mask_name.split('.')[0]
        #img_second = img_first.split('_mask')[0]      
        #image_name =img_second+'.jpg'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        gray_img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        #img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        acc_imgs[i] = acc_img
        gray_mgs[i] = gray_img
        #imgs_mask[i] = img_mask

        i += 1
        print ('Done',i)
    
    return acc_imgs, gray_mgs

def read_images_and_masks(data_path, image_h, image_w):
    
    train_data_path = os.path.join(data_path)
    images = glob.glob(train_data_path + "/*mask.png")
    total = np.round(len(images)) 

    acc_imgs = np.ndarray((total, image_h, image_w,3), dtype=np.uint8)
    imgs = np.zeros((total, image_h, image_w), dtype=np.uint8)
    imgs_mask = np.zeros((total, image_h, image_w), dtype=np.uint8)
    
    i = 0
    print('Creating training images...')
    #img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:

        image_mask_name = image_name.split('/')[-1]      
        img_first = image_mask_name.split('.')[0]
        img_second = img_first.split('_mask')[0]      
        image_name =img_second+'.png'
                     
        acc_img = cv2.imread(os.path.join(train_data_path, image_name))
        img = cv2.imread(os.path.join(train_data_path, image_name),cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        acc_imgs[i] = acc_img
        imgs[i] = img
        imgs_mask[i] = img_mask

        i += 1
        print ('Done',i)
    
    return acc_imgs,imgs,imgs_mask


def read_traning_data_4classificaiton(base_dir, h,w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            #print(img_name_1)
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='jpg' or 'png':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                X.append(img)
                y.append(class_index)
                
                useful_image_count += 1
        

    X = np.array(X).astype(np.float32)
    #X = X.transpose((0, 3, 1, 2))
    X=X.transpose((0,1,2,3))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags

def read_traning_data_4classificaiton_camelyon16(base_dir, h,w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            #print(img_name_1)
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='png' or 'jpg':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                
                if class_index ==0:
                    X.append(img)
                    y.append(class_index)
                    useful_image_count += 1
                else:
                    X.append(img)
                    y.append(class_index)
                
                    flip_lr = np.fliplr(img)
                    X.append(flip_lr)
                    y.append(class_index)
                    
                    flip_ud = np.flipud(img)
                    X.append(flip_ud)
                    y.append(class_index)
                    
                    useful_image_count += 3
        

    X = np.array(X).astype(np.float32)
    #X = X.transpose((0, 3, 1, 2))
    X=X.transpose((0,1,2,3))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags

def read_traning_data_4classificaiton_camelyon16_hne(base_dir, h,w):
        
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)
      
    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    
    #pdb.set_trace()
    
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
         
            img = cv2.imread(filename)
            img_name_1 = filename.split('/')[-1]
            #print(img_name_1)
            img_name = img_name_1.split('.')[0]
            img_extension = img_name_1.split('.')[1]
            
            if img_extension =='png' or 'jpg':
                img= np.array(img)               
                img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
                X.append(img)
                y.append(class_index)
                
                flip_lr = np.fliplr(img)
                X.append(flip_lr)
                y.append(class_index)
                    
                flip_ud = np.flipud(img)
                X.append(flip_ud)
                y.append(class_index)
                    
                useful_image_count += 3
        

    X = np.array(X).astype(np.float32)
    #X = X.transpose((0, 3, 1, 2))
    X=X.transpose((0,1,2,3))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print("classes:")
    for class_index, class_name in enumerate(tags):
        print(class_name, sum(y == class_index))
    
    print("\n")

    return X, y, tags