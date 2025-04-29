# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:39:35 2019

@author: deeplens
"""

import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf

import time, datetime

#kernel = np.ones((5,5), np.uint8) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    # st = time.time()
    # w = label.shape[0]
    # h = label.shape[1]
    # num_classes = len(class_dict)
    # x = np.zeros([w,h,num_classes])
    # unique_labels = sortedlist((class_dict.values()))
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index = unique_labels.index(list(label[i][j][:]))
    #         x[i,j,index]=1
    # print("Time 1 = ", time.time() - st)

    # st = time.time()
    # https://stackoverflow.com/questions/46903885/map-rgb-semantic-maps-to-one-hot-encodings-and-vice-versa-in-tensorflow
    # https://stackoverflow.com/questions/14859458/how-to-check-if-all-values-in-the-columns-of-a-numpy-matrix-are-the-same
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map


def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''

    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)

    # If you need to invert the black/white data image
    # im_blur = np.invert(im_blur)
    # Convert back to BGR for cv2
    #im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)

    # Apply colormap
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)

    # blend images 50/50
    return (a1*im_map + a2*im_cloud_clr).astype(np.uint8) 
    
def perform_morphological_operations(pred):
    
    #pred_mask = cv2.erode(pred,kernel,iterations = 1)

    pred_mask = cv2.dilate(pred,kernel,iterations = 1)            
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)    #  To fillup the internal pixels...
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)     # opening operation to remove the noise 
    pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)

    #pred_mask = cv2.dilate(pred_mask,kernel,iterations = 1)       
    #pred_mask = cv2.erode(pred_mask,kernel,iterations = 1)  
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
     
    pred = pred_mask        
    r,c = pred_mask.shape       
    pred_mask[0:1,:]=0
    pred_mask[r-1:r,:]=0    
    pred_mask[:,0:1]=0
    pred_mask[:,c-1:c]=0 
    
    return pred_mask
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# class_dict = get_class_dict("CamVid/class_dict.csv")
# gt = cv2.imread("CamVid/test_labels/0001TP_007170_L.png",-1)
# gt = reverse_one_hot(one_hot_it(gt, class_dict))
# gt = colour_code_segmentation(gt, class_dict)

# file_name = "gt_test.png"
# cv2.imwrite(file_name,np.uint8(gt))

def HSV_image_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    h = np.argmax(image, axis = -1)
    v = np.amax(image, axis = -1) * 255
    s = np.ones(h.shape)*255
    colour_codes = 255 / len(label_values)
    h = h * colour_codes
    h = np.expand_dims(h, axis=2)
    s = np.expand_dims(s, axis=2)
    v = np.expand_dims(v, axis=2)
    print(h.shape)
    x = np.concatenate([h, s, v],2)
    

    return x
