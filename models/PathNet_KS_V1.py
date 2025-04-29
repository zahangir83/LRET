#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:08:50 2020

@author: malom
"""
import keras
#from keras_applications import vgg16
#import keras_applications
#from keras.applications.imagenet_utils import _obtain_input_shape
#from keras.applications.imagenet_utils import obtain_input_shape
from keras.applications import vgg16
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
#from keras_vggface import utils
#from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import get_source_inputs
import warnings
from keras.models import Model
from keras import layers

import pdb




def PathNet_VGG19(include_top=True, input_shape=None, pooling=None, classes=74):

    input_tensor = keras.Input(shape=input_shape)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        
    # Block 1
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1_1')(
        img_input)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    
    # Block 6
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_1')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_2')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool6')(x)
    
    # Block 7
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv7_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv7_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv7_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool7')(x)

    # Block 8
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8_2')(
         x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8_3')(
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool8')(x)
    
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PathNet_VGG16')
    
   
    return model

def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
        shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def PathNet_RESNET50(include_top=True, input_shape=None, pooling=None, classes=74):
    

    input_tensor = keras.Input(shape=input_shape)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # General convolutional layer
    x = Conv2D(4, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #pdb.set_trace()


    x = resnet_conv_block(x, 3, [8, 8, 16], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [8, 8, 16], stage=2, block=2)
    x = resnet_identity_block(x, 3, [8, 8, 16], stage=2, block=3)

    x = resnet_conv_block(x, 3, [16, 16, 32], stage=3, block=1)
    x = resnet_identity_block(x, 3, [16, 16, 32], stage=3, block=2)
    x = resnet_identity_block(x, 3, [16, 16, 32], stage=3, block=3)
    x = resnet_identity_block(x, 3, [16, 16, 32], stage=3, block=4)

    x = resnet_conv_block(x, 3, [32, 32, 64], stage=4, block=1)
    x = resnet_identity_block(x, 3, [32, 32, 64], stage=4, block=2)
    x = resnet_identity_block(x, 3, [32, 32, 64], stage=4, block=3)
    x = resnet_identity_block(x, 3, [32, 32, 64], stage=4, block=4)
    x = resnet_identity_block(x, 3, [32, 32, 64], stage=4, block=5)
    x = resnet_identity_block(x, 3, [32, 32, 64], stage=4, block=6)

    x = resnet_conv_block(x, 3, [64, 64, 128], stage=5, block=1)
    x = resnet_identity_block(x, 3, [64, 64, 128], stage=5, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 128], stage=5, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 256], stage=6, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 256], stage=6, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 256], stage=6, block=3)
     
    x = resnet_conv_block(x, 3, [256, 256, 512], stage=7, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 512], stage=7, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 512], stage=7, block=3)
    
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PathNet_resnet50')

    return model



if __name__ == '__main__':
    
    
    input_shape = (1024,1024,3)
    pooling='avg'
    classes=74
    include_top=True 
    pdb.set_trace()
    model = PathNet_RESNET50(input_shape=input_shape, pooling=None, classes=classes)
    
    model.summary()
    

    
