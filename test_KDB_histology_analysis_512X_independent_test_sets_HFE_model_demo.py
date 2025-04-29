#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pdb
from tensorflow.keras import layers
from tensorflow import keras
import argparse
from os.path import join as join_path
import matplotlib.pyplot as plt
import pathlib
from models import LR_models
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import backend as K
from skimage.transform import resize

from skimage import io
import random
import cv2
import pandas as pd

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   #fname='flower_photos',
                                   #untar=True)
import json                                   
print(tf.__version__)
INTERPOLATION = "bilinear"

AUTO = tf.data.AUTOTUNE
#EPOCHS = 150
image_size = (512, 512)

abspath = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def most_frequent(List): 
    return max(set(List), key = List.count) 
  
def random_number_generator_in_range(number_samples, how_many_samples, top_range):
    random_vector = []
    for k in range(number_samples):
         rand_samples = random.sample(range(0,top_range),how_many_samples)
         random_vector.append(rand_samples)
    return random_vector

# Feature maps layer name: conv5_3_1x1_increase/bn  / activation_48
# Clustering feature maps:  onv7_3_1x1_increase/bn  /  activation_66 / avg_pool / 
class FeatureHeatmapGenerator:


    def __init__(self, model_path, alpha=0.6, conv_name_heatmap='FEXT_7x7x512'):
        model_4gpus = load_model(model_path)
        self.model = model_4gpus #.layers[-2]
        self.alpha = alpha
        self.conv_name_hm = conv_name_heatmap
        self.conv_name_1D = 'FEXL_512'
        self.conv_name_2D = 'FEXT_7x7x512'
        #self.conv_name_2D_from_BB = 'relu' #'conv5_block16_concat'



    def heatmap_of_feature_from_dirs_save_same_dirs(self, scr_dir, feature_number):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
     
        actual_labels = []
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)     
                actual_labels.append(dir_name)
                    
                image_list = os.listdir(sub_dir_path) # dir is your directory path
                total_samples = len(image_list)
                images_files_final = np.array(image_list)
                print('Totla number of samples:', total_samples)                   

                for i, img_name in enumerate(images_files_final): 
                    #print('Copying files for : ',dir_name )
                    image_path = join_path(sub_dir_path,img_name)   
                    print(image_path)
                    img_ext = img_name.split('.')[1]                
                    #pdb.set_trace()
                    if img_ext =='jpg':                  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            img = cv2.imread(image_path, cv2.IMREAD_COLOR)   
                            img_name_owext = img_name.split('.')[0]
                            img_name_4s = img_name_owext+'_heatmap.png'
                            cp_np_name = img_name_owext+'_class_prob.npy'
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            
                            # Generate heatmap and extract features and class probs.
                            fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
                            # save all representations...
                            final_dst_image = join_path(sub_dir_path, img_name)
                            final_dst_heatmap = join_path(sub_dir_path, img_name_4s)
                            final_class_prob_saving_path = join_path(sub_dir_path, cp_np_name)
                            final_fm_saving_path = join_path(sub_dir_path, fm_np_name)

                            cv2.imwrite(final_dst_image, img)
                            cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)


        actual_labels = np.array(actual_labels)
        np.save(os.path.join(sub_dir_path,'y_ac_labels.npy'), actual_labels)
        pd.DataFrame(actual_labels).to_csv(os.path.join(sub_dir_path,'y_ac_labels.csv'))     

    def heatmap_of_feature_from_dirs_save_diff_dirs(self, scr_dir, feature_number, save_dir):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
     
        actual_labels = []
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)  
                
                actual_labels.append(dir_name)
                
                if not os.path.isdir("%s/%s"%(save_dir,dir_name)):
                    os.makedirs("%s/%s"%(save_dir,dir_name))                
                dst_dir_final = join_path(save_dir,dir_name+'/')    
                
                image_list = os.listdir(sub_dir_path) # dir is your directory path
                total_samples = len(image_list)
                images_files_final = np.array(image_list)
                print('Totla number of samples:', total_samples)                   

                for i, img_name in enumerate(images_files_final): 
                    #print('Copying files for : ',dir_name )
                    image_path = join_path(sub_dir_path,img_name)   
                    print(image_path)
                    img_ext = img_name.split('.')[1]                
                    #pdb.set_trace()
                    if img_ext =='png':                  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            # img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            # img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)
                            img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
                            #img_array = keras.preprocessing.image.img_to_array(img)
                            #img_array = tf.expand_dims(img_array, 0)  # Create batch axis        
                            
                            img_name_owext = img_name.split('.')[0]
                            img_name_4s = img_name_owext+'_heatmap.png'
                            cp_np_name = img_name_owext+'_class_prob.npy'
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            
                            # Generate heatmap and extract features and class probs.
                            
                            fam, conv_output_fm, class_probs = self._heatmap_of_feature_helper(image_path, feature_number)
                            
                            #fam, conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                            # save all representations...
                            final_dst_image = join_path(dst_dir_final, img_name)
                            final_dst_heatmap = join_path(dst_dir_final, img_name_4s)
                            final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)
                            final_fm_saving_path = join_path(dst_dir_final, fm_np_name)

                            cv2.imwrite(final_dst_image, img)
                            cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)
        
        actual_labels = np.array(actual_labels)
        np.save(os.path.join(save_dir,'y_ac_labels.npy'), actual_labels)
        pd.DataFrame(actual_labels).to_csv(os.path.join(save_dir,'y_ac_labels.csv'))                         


    def class_activation_heatmap_of_feature_from_dirs_save_diff_dirs(self, scr_dir, feature_number, save_dir):
        '''
        Saves the feature heatmap of the given image

        :param image_path:im
        :param feature_number:
        :param save_dir:
        :return:
        '''
     
        actual_labels = []
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)  
                
                actual_labels.append(dir_name)
                
                if not os.path.isdir("%s/%s"%(save_dir,dir_name)):
                    os.makedirs("%s/%s"%(save_dir,dir_name))                
                dst_dir_final = join_path(save_dir,dir_name+'/')    
                
                image_list = os.listdir(sub_dir_path) # dir is your directory path
                total_samples = len(image_list)
                images_files_final = np.array(image_list)
                print('Totla number of samples:', total_samples)                   

                for i, img_name in enumerate(images_files_final): 
                    #print('Copying files for : ',dir_name )
                    image_path = join_path(sub_dir_path,img_name)   
                    print(image_path)
                    img_ext = img_name.split('.')[1]
                    
                    #pdb.set_trace()
                    
                    #if img_ext =='png': 
                    
                    if img_ext =='jpg':  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            # img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            # img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)
                            img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
                            #img_array = keras.preprocessing.image.img_to_array(img)
                            #img_array = tf.expand_dims(img_array, 0)  # Create batch axis        
                            
                            img_name_owext = img_name.split('.')[0]
                            img_name_4s = img_name_owext+'_heatmap.png'
                            class_actiation_img_name_4s = img_name_owext+'_heatmap_class_act.png'
                            
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            cp_np_name = img_name_owext+'_class_prob.npy'

                            
                            # Generate heatmap and extract features and class probs.
                            
                            fam, fam_class_activation, conv_output_fm, class_probs = self._class_activation_heatmap_of_feature_helper(image_path, feature_number)
                            
                            #fam, conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                            # save all representations...
                            final_dst_image = join_path(dst_dir_final, img_name)
                            final_dst_heatmap = join_path(dst_dir_final, img_name_4s)
                            final_dst_heatmap_class_activation = join_path(dst_dir_final, class_actiation_img_name_4s)

                            final_fm_saving_path = join_path(dst_dir_final, fm_np_name)
                            final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)

                            #cv2.imwrite(final_dst_image, img)
                            cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                            cv2.imwrite(final_dst_heatmap_class_activation, fam_class_activation.astype(np.uint8, copy=False))
                            
                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)
        
        # actual_labels = np.array(actual_labels)
        # np.save(os.path.join(save_dir,'y_ac_labels.npy'), actual_labels)
        # pd.DataFrame(actual_labels).to_csv(os.path.join(save_dir,'y_ac_labels.csv'))                
        
    def class_activation_heatmap_of_feature_from_dirs_save_diff_dirs_256X(self, scr_dir, feature_number, save_dir):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
     
        actual_labels = []
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)  
                
                actual_labels.append(dir_name)
                
                if not os.path.isdir("%s/%s"%(save_dir,dir_name)):
                    os.makedirs("%s/%s"%(save_dir,dir_name))                
                dst_dir_final = join_path(save_dir,dir_name+'/')    
                
                image_list = os.listdir(sub_dir_path) # dir is your directory path
                total_samples = len(image_list)
                images_files_final = np.array(image_list)
                print('Totla number of samples:', total_samples)                   

                for i, img_name in enumerate(images_files_final): 
                    #print('Copying files for : ',dir_name )
                    image_path = join_path(sub_dir_path,img_name)   
                    print(image_path)
                    img_ext = img_name.split('.')[1]                
                    #pdb.set_trace()
                    if img_ext =='png':                  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            # img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            # img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)
                            img = keras.preprocessing.image.load_img(image_path, target_size=(256,256))
                            #img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                            #img_array = keras.preprocessing.image.img_to_array(img)
                            #img_array = tf.expand_dims(img_array, 0)  # Create batch axis        
                            
                            img_name_owext = img_name.split('.')[0]
                            img_name_4s = img_name_owext+'_heatmap.png'
                            class_actiation_img_name_4s = img_name_owext+'_heatmap_class_act.png'
                            
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            cp_np_name = img_name_owext+'_class_prob.npy'

                            
                            # Generate heatmap and extract features and class probs.
                            
                            fam, fam_class_activation, conv_output_fm, class_probs = self._class_activation_heatmap_of_feature_helper(image_path, feature_number)
                            
                            #fam, conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                            # save all representations...
                            final_dst_image = join_path(dst_dir_final, img_name)
                            final_dst_heatmap = join_path(dst_dir_final, img_name_4s)
                            final_dst_heatmap_class_activation = join_path(dst_dir_final, class_actiation_img_name_4s)

                            final_fm_saving_path = join_path(dst_dir_final, fm_np_name)
                            final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)

                            #cv2.imwrite(final_dst_image, img)
                            cv2.imwrite(final_dst_heatmap, fam.astype(np.uint8, copy=False))
                            cv2.imwrite(final_dst_heatmap_class_activation, fam_class_activation.astype(np.uint8, copy=False))
                            
                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)
        
        # actual_labels = np.array(actual_labels)
        # np.save(os.path.join(save_dir,'y_ac_labels.npy'), actual_labels)
        # pd.DataFrame(actual_labels).to_csv(os.path.join(save_dir,'y_ac_labels.csv'))     
    def feature_extraction_class_probs_from_dirs_save_diff_dirs(self, scr_dir, feature_number, save_dir):
        '''
        Saves the feature heatmap of the given image

        :param image_path:
        :param feature_number:
        :param save_dir:
        :return:
        '''
        actual_labels = []
        pdb.set_trace()
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
                print(dir_name)      
                if not os.path.isdir("%s/%s"%(save_dir,dir_name)):
                    os.makedirs("%s/%s"%(save_dir,dir_name))                
                dst_dir_final = join_path(save_dir,dir_name+'/')    
                
                actual_labels.append(dir_name)
                
                image_list = os.listdir(sub_dir_path) # dir is your directory path
                total_samples = len(image_list)
                images_files_final = np.array(image_list)
                print('Totla number of samples:', total_samples)                   

                for i, img_name in enumerate(images_files_final): 
                    #print('Copying files for : ',dir_name )
                    image_path = join_path(sub_dir_path,img_name)   
                    print(image_path)
                    #img_ext = img_name.split('.')[1]     
                    img_ext = img_name.split('.')[-1]  ## Only for testing case of LGG vs GBM.......
                    #pdb.set_trace()
                    if (img_ext =='png' or img_ext =='jpg'):                  
                        check_valid_image_flag = verify_image(image_path)                    
                        if check_valid_image_flag == True:                            
                            #img = cv2.imread(image_path, cv2.IMREAD_COLOR)  
                            #img = cv2.resize(img, image_size, interpolation = cv2.INTER_AREA)
                            #img_name_owext = img_name.split('.')[0]   # General cases ..(all)
                            
                            img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
                            img_array = keras.preprocessing.image.img_to_array(img)
                            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

                            
                            img_name_owext = img_name.split('.')[0]+'.'+img_name.split('.')[1] ## Only for LGG s GBM testing .....
                            cp_np_name = img_name_owext+'_class_prob.npy'
                            fm_np_name = img_name_owext+'_fm_vector.npy'
                            
                            # Generate heatmap and extract features and class probs.
                            conv_output_fm, class_probs = self.feature_extract_class_prob_helper(image_path, feature_number)
                            # save all representations...
                            final_class_prob_saving_path = join_path(dst_dir_final, cp_np_name)
                            final_fm_saving_path = join_path(dst_dir_final, fm_np_name)

                            np.save(final_fm_saving_path,conv_output_fm)
                            np.save(final_class_prob_saving_path,class_probs)
        
        # actual_labels = np.array(actual_labels)
        # np.save(os.path.join(save_dir,'y_ac_labels.npy'), actual_labels)
        # pd.DataFrame(actual_labels).to_csv(os.path.join(save_dir,'y_ac_labels.csv'))     
        
                            
    def feature_extract_class_prob_helper(self, image_path, feature_number):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''
        
        #image = cv2.imread(image_path)
        #image = cv2.resize(image, (768,768), interpolation = cv2.INTER_AREA)   
        
        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        final_conv_layer_2D = self.model.get_layer(name=self.conv_name_2D)
        get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_2D.output])
        
        # #>>>>>>>>>>>>>>>>>>>>>> EXACT FEATURES FOR 1K <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # layer_dict = dict([(layer.name,layer) for layer in self.model.layers])
        # layer_name_1D = self.conv_name_1D
        # features_1D = Model(inputs=self.model.inputs,outputs=layer_dict[layer_name_1D].output)  
        # layer_name_2D = self.conv_name_2D
        # features_2D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_2D].output)
    
        features_2D = np.squeeze(get_output_fm(img_array)[0]) 
        features_2D = np.squeeze(features_2D)
        # >>>>>>>>>>>>>>>>>>>>>. Feature extractors <<<<<<<<<<<<<<<<<<<<<<<<<<<

        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        class_probs = self.model.predict(img_array)  
        
        #return features_1D,features_2D, class_probs
        return features_2D, class_probs

    
    def _heatmap_of_feature_helper(self, image_path, feature_number):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''
        
        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array_in = keras.preprocessing.image.img_to_array(img)        
        width, height, _ = img_array_in.shape   
        img_array = tf.expand_dims(img_array_in, 0)  # Create batch axis

        #for layer in self.model.layers:
        #   print(layer.output_shape)
         #  print(layer.name)
        final_conv_layer = self.model.get_layer(name=self.conv_name_2D)
        get_output = K.function([self.model.layers[0].input], [final_conv_layer.output])
        conv_output = np.squeeze(get_output(img_array)[0])    

        #h,w, feature_number =   conv_output.shape        
        # generate heatmap
        cam = conv_output[:, :, feature_number - 1]
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image
        heatmap_final = cv2.addWeighted(img_array_in, self.alpha, heatmap, 1 - self.alpha, 0)
        # xxxxxxxxx  Heatmap section is done here   xxxxxxxxxxxxxx     
        
        # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        final_conv_layer_2D = self.model.get_layer(name=self.conv_name_2D)
        get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_2D.output])
        conv_output_fm = np.squeeze(get_output_fm()[0]) 
        conv_output_fm = np.squeeze(conv_output_fm)
        # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        #pdb.set_trace()
        #dl_input = (img_array_in / 255).astype(float)  
        class_probs = self.model.predict(img_array)  
        
        return heatmap_final , conv_output_fm, class_probs
    
    def _class_activation_heatmap_of_feature_helper(self, image_path, feature_number):
        '''
        Returns the passed in image with a heatmap of the specific feature number overlayed
        :param model_path:
        :param image_path:
        :param feature_number:
        :param conv_name:
        :return:
        '''

        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array_in = keras.preprocessing.image.img_to_array(img)        
        width, height, _ = img_array_in.shape   
        
        #img_array_in = img_array_in/255
        img_array = tf.expand_dims(img_array_in, 0)  # Create batch axis

        #pdb.set_trace()
        #for layer in self.model.layers:
        #   print(layer.output_shape)
         #  print(layer.name)
        final_conv_layer_2D = self.model.get_layer(name=self.conv_name_2D)
        get_output = K.function([self.model.layers[0].input], [final_conv_layer_2D.output])
        conv_output = np.squeeze(get_output(img_array)[0])    
        #features_2D = np.squeeze(conv_output)
        #h,w, feature_number =   conv_output.shape        
        # generate heatmap
        #cam = conv_output[:, :, feature_number - 1]
        cam = conv_output.mean(axis=2)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # overlay heatmap on original image
        cv2.imwrite('test.png',heatmap.astype(np.uint8,copy=False))
        
        heatmap_final = cv2.addWeighted(np.uint8(img_array_in), self.alpha, heatmap, 1 - self.alpha, 0)#,dtype=cv2.CV_32F)
        # xxxxxxxxx  Heatmap section is done here   xxxxxxxxxxxxxx
        #cv2.imwrite('test_with_input.png',heatmap_final.astype(np.uint8,copy=False))
        
        # # >>>>>>>>>   Extracted feature maps from here<<<<<<<<<<<<<<<<< 
        # final_conv_layer_2D = self.model.get_layer(name=self.conv_name_2D)
        # get_output_fm = K.function([self.model.layers[0].input], [final_conv_layer_2D.output])
        # features_2D = np.squeeze(get_output_fm(img_array)[0]) 
        # features_2D = np.squeeze(features_2D)
        # # xxxxxxxxx  End of feature maps extraction here   xxxxxxxxxxxxxx    
        
        # >>>>>>>>>  check the class probability.....
        #pdb.set_trace()
        #dl_input = (img_array_in / 255).astype(float)  
        class_probs = self.model.predict(img_array)  
        print('Class probs :', class_probs)
        # Get prediction
        class_of_interest = np.argmax(class_probs)   
        print('Class of interest: ', class_of_interest)
        # >>>>>>>>>>>>>>>>>>>>>>> GENERATE THE CLASS SPECIFIC HEATMAP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
     
        dense_weights = self.model.layers[-1].get_weights()[0]
        # Remove dimensions with only one entry from array shape
        last_conv_layer = np.squeeze(conv_output)  
        
        # Get weights of dense layer for class_of_interest
        class_of_interest = 2
        dense_weights_curent_class = dense_weights[:, class_of_interest]
        #dense_weights_curent_class_exd = np.expand_dims(np.array(dense_weights_curent_class),axis=-1)
        # Dot product of last conv layer (8,8,2048) with dense_weights_current_class (2048,,)
        class_activation_map = np.dot(last_conv_layer, dense_weights_curent_class)
        mxn_class_activation_map = (class_activation_map - class_activation_map.min())/(class_activation_map.max() - class_activation_map.min())
        ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. THRESHOLDING ON THE CLASS ACTIVATION <<<<<<<<<<<<<<<<<<<<<<<<<<
        # mxn_class_activation_map
        # Normalize to -1 1 and rescale to 299, 299
        class_activation_map_resized = cv2.resize(mxn_class_activation_map, (width, height))
        
        heatmap_class_activation = cv2.applyColorMap(np.uint8(255 * class_activation_map_resized), cv2.COLORMAP_JET)
        cv2.imwrite('test_class_act.png',heatmap_class_activation.astype(np.uint8,copy=False))
        
        class_activation_heatmap_final = cv2.addWeighted(np.uint8(img_array_in), self.alpha, heatmap_class_activation, 1 - self.alpha, 0)#,dtype=cv2.CV_32F)

        
        return heatmap_final, class_activation_heatmap_final, last_conv_layer, class_probs
    
    
    
    def read_feature_and_pred_and_merge_them(self, scr_dir):
        case_ids = []
        y_gt = []
        y_pred = []
        fts_mat = []
    
        #pdb.set_trace()
        
        for path, subdirs, files in os.walk(scr_dir):        
            for dir_name in subdirs:            
                sub_dir_path = os.path.join(scr_dir, dir_name+'/')           
    
                #file_list = os.listdir(sub_dir_path) # dir is your directory path
                file_list = [x for x in sorted(os.listdir(sub_dir_path)) if x[-14:] == '_fm_vector.npy']   
                total_samples = len(file_list)
                files_final = np.array(file_list)
                #print('Totla number of samples:', total_samples) 
    
                for i, file_name in enumerate(files_final): 
                    file_path = os.path.join(sub_dir_path,file_name)   
                    #print(file_path)
                    file_ext = file_name.split('.')[1]      
                    file_name_wo_ext = file_name.split('_fm_vector')[0]#+'.'+file_name.split('.')[1] 
                    file_name_wo_ext_fp = file_name.split('_fm_vector')[0]#+'.'
    
    
                    #if file_name.endswith('_class_prob.npy'):
                        
                    #    case_ids.append(file_name_wo_ext)
                    #    y_gt.append(dir_name)
                    
                    #    input_pred = np.load(file_path,allow_pickle=True)
                    #    y_pred.append(input_pred)
    
                    if file_name.endswith('_fm_vector.npy'):
                        
                        case_ids.append(file_name_wo_ext)
                        y_gt.append(dir_name)
                        # read the DL feature representation.....
                        input_fv = np.load(file_path,allow_pickle=True)
                        input_fv_0 = np.mean(input_fv,axis=0)
                        input_fv_00 = np.mean(input_fv_0,axis=0)
                        exd_input_fv_00 = np.expand_dims(input_fv_00,axis=0)
                        fts_mat.append(exd_input_fv_00)
                        
                        # read the predicted probability..........
                        
                        file_path_pred = os.path.join(sub_dir_path,file_name_wo_ext_fp+'_class_prob.npy') 
                        input_pred = np.load(file_path_pred,allow_pickle=True)
                        y_pred.append(input_pred)
        
        
    
        case_ids = np.array(case_ids)
        y_gt = np.array(y_gt)
        y_pred_mat = np.squeeze(np.array(y_pred ))
        fts_mat = np.squeeze(np.array(fts_mat)) 
        
        return case_ids,y_gt,y_pred_mat,fts_mat
                    


        
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="KDB_Path_DenseNet121_100EPS_2P100GPUs", help='Name of your project')
parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')

#parser.add_argument('--dataset_path_test', type=str, default="/home/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/Kather_DB/train_val_test_DB_512x/train/", help='Dataset you are using.')
                                                             
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>.    IN TESTING DATA PATH <<<<<<<<<<<<<<<<<<<<<<<<<<<
parser.add_argument('--dataset_path_test', type=str, default="/home/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/LC_KDB_testing_from_74CLSDB/lung_512x_patches/", help='Dataset you are using.')
#parser.add_argument('--dataset_path_test', type=str, default="/home/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/LC_KDB_testing_from_74CLSDB/adipose_512x_patches/adipose/", help='Dataset you are using.')

parser.add_argument('--input_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--input_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--target_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--target_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--num_class', type=int, default=3, help='Number of images in each batch')

parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
#parser.add_argument('--model', type=str, default="R2UNet_DP", help='The model you are using. See model_builder.py for supported models')
args = parser.parse_args()

INP_SIZE = (args.input_height, args.input_width)
TARGET_SIZE = (args.target_height, args.target_width)
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
num_classes = args.num_class
 
#pdb.set_trace()
if not os.path.isdir("%s/%s"%("experimental_logs",args.project_name)):
    os.makedirs("%s/%s"%("experimental_logs",args.project_name))

project_path = join_path('experimental_logs/', args.project_name+'/')
if not os.path.isdir("%s/%s"%(project_path,"training")):
    os.makedirs("%s/%s"%(project_path,"training"))
    os.makedirs("%s/%s"%(project_path,"testing"))
    os.makedirs("%s/%s"%(project_path,"weights"))
    os.makedirs("%s/%s"%(project_path,"trained_model"))
# create all necessary path for saving log files
    
training_log_saving_path = join_path(project_path,'training/')
testing_log_saving_path = join_path(project_path,'testing/')
#weight_saving_path = join_path(project_path,'weights/')
traned_model_saving_path = join_path(project_path,'trained_model/')

# =============================================================================

#pdb.set_trace()

checkpoint_filepath = os.path.join(traned_model_saving_path,args.project_name+'_'+str(INP_SIZE[0])+'x'+str(INP_SIZE[0])+'_best_model'+'.h5')
#final_model_sv_path = os.path.join(traned_model_saving_path,'final_model.h5')
feature_number = 32   # number of feature maps considered for generating heatmaps...512(8)/256(16)/128(32)/64(64)
fhg = FeatureHeatmapGenerator(checkpoint_filepath)

# Image path and extracted features for LGG_vs_GBM survival analysis project........
data_dir_test = pathlib.Path(args.dataset_path_test)

in_out_OOD_path = 'test_independent_Lung_Adenocarcinoma'

if not os.path.isdir("%s/%s"%(testing_log_saving_path,in_out_OOD_path)):
     os.makedirs("%s/%s"%(testing_log_saving_path,in_out_OOD_path))
out_dir =  join_path(testing_log_saving_path,in_out_OOD_path)

#pdb.set_trace()  
#fhg.feature_extraction_class_probs_from_dirs_save_diff_dirs(data_dir_test, feature_number, out_dir)
fhg.class_activation_heatmap_of_feature_from_dirs_save_diff_dirs(data_dir_test, feature_number, out_dir) # with featue anad class specific heatmaps.....

### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MERGED extracted feature representation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
case_ids_train_in,y_gt_train_in,y_pred_mat_train_in,fts_mat_train_in = fhg.read_feature_and_pred_and_merge_them(out_dir)

print('Num. cases :', case_ids_train_in.shape)
print('GT shape :', y_gt_train_in.shape)
print('Pred shape :', y_pred_mat_train_in.shape)
print('Feature shape :', fts_mat_train_in.shape)

np.save(os.path.join(out_dir,'case_ids_'+'test.npy'),case_ids_train_in)
np.save(os.path.join(out_dir,'y_ac_'+'test.npy'),y_gt_train_in)
np.save(os.path.join(out_dir,'y_pred_'+'test.npy'),y_pred_mat_train_in)
np.save(os.path.join(out_dir,'EF_'+'test.npy'),fts_mat_train_in)



#fhg.heatmap_of_feature_from_dirs_save_diff_dirs(data_dir_test, feature_number, out_dir)  # only useing IN TEST SAMPLES <<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>....  Refrence codes perfectly working for me <<<<<<<<<<<<<<<<<<<<<<<<<<<<,,  .....................

# # model loading and testing ........
# model = load_model(checkpoint_filepath)
# print('Summary of the model :')
# model.summary()

# #>>>>>>>>>>>>>>>>>>>>>> EXACT FEATURES FOR 1K <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# layer_dict = dict([(layer.name,layer) for layer in model.layers])
# layer_name_1D = 'FEXL_512'
# feature_extractor_1D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_1D].output)
# layer_name_2D = 'FEXT_7x7x512'
# feature_extractor_2D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_2D].output)
# # >>>>>>>>>>>>>>>>>>>>>. Feature extractors <<<<<<<<<<<<<<<<<<<<<<<<<<<


# # Read the testing dataset from sub-directory and test with the model....
# testing_data_dir = args.dataset_path_test
# idv_img_path = os.path.join(args.dataset_path_test,'lung_aca/lungaca1007.png')

# img = keras.preprocessing.image.load_img(
#     idv_img_path, target_size=image_size
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# predictions = model.predict(img_array)
# score = float(predictions[0])
# print("Predicted score is : ", score)

# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.   Extracted features     <<<<<<<<<<<<<<<<<<<<<<<<
# x_test_feature = feature_extractor_2D.predict(img_array)
# x_test_feature = np.array(x_test_feature)



                            

