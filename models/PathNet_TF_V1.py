#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:08:50 2020

@author: malom
"""
import tensorflow as tf
from tensorflow import keras
#from keras_applications import vgg16
#import keras_applications
#from keras.applications.imagenet_utils import _obtain_input_shape
#from keras.applications.imagenet_utils import obtain_input_shape
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
    
from tensorflow.keras.layers import ZeroPadding2D
#from tensorflow.keras.utils import layer_utils
#from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
#from keras_vggface import utils
#from keras.engine.topology import get_source_inputs
#from tensorflow.keras.utils.layer_utils import get_source_inputs
#from tensorflow.keras.utils.layer_utils import get_source_inputs
#from tensorflow.keras.engine.topology import get_source_inputs
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import pdb

TARGET_SIZE_WMXP = (255, 255)

TARGET_SIZE_ResNet50 = (255, 255)

TARGET_SIZE = (256, 256)
TARGET_SIZE_CXN = (256, 256)
TARGET_SIZE_512X = (262, 262)
TARGET_SIZE_DenseNet = (255, 255)

TARGET_SIZE_WOMP = (256, 256)

TARGET_SIZE_WOMP_LCDB = (384, 384)
TARGET_SIZE_WMP_LCDB = (192, 192)


INTERPOLATION = "bilinear"

def conv_block(x, filters, kernel_size, strides, activation=layers.LeakyReLU(0.2)):
    x = layers.Conv2D(filters, kernel_size, strides, padding="same", use_bias=False)(x)
    if activation:
        x = activation(x)
    x = layers.BatchNormalization()(x)
    return x


# def res_block(x):
#     inputs = x
#     x = conv_block(x, 16, 3, 1)
#     x = conv_block(x, 16, 3, 1, activation=None)
#     return layers.Add()([inputs, x])

def get_HDFE(inputs, filters=3):

    # We first need to resize to a fixed resolution to allow mini-batch learning
    #naive_resize = layers.experimental.preprocessing.Resizing(*TARGET_SIZE,
    #        interpolation=interpolation)(inputs)

    # First conv block without Batch Norm
    x = layers.Conv2D(filters=3, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # Second conv block with Batch Norm
    
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    
    x = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)




    # >>>>>>>>>>>>>>>>>>>>... REplacate <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # General convolutional layer
    
    # bn_axis = 3
    # x = Conv2D(4, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/7x7_s2')(x)
    # x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_1/3x3_b2')(x)
    # x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_2/3x3_b2')(x)
    
    return x


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


    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='classifier')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    
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


def PathNet_RESNET50(INP_SIZE, num_classes):
    

    # input_tensor = keras.Input(shape=input_shape)
    
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # if K.image_data_format() == 'channels_last':
    bn_axis = 3
    # else:
    #     bn_axis = 1

    #pdb.set_trace()

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    
    # General convolutional layer
    x = Conv2D(4, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/7x7_s2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_1/3x3_b2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_2/3x3_b2')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_1/3x3_b3')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_2/3x3_b3')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_3/3x3_b3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)


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
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='classifier')(x)

    #inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PathNet_resnet50')

    return model



def PathNet_RESNET50_AVGP(INP_SIZE, num_classes):
    

    # input_tensor = keras.Input(shape=input_shape)
    
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # if K.image_data_format() == 'channels_last':
    bn_axis = 3
    # else:
    #     bn_axis = 1

    #pdb.set_trace()

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    
    # General convolutional layer
    x = Conv2D(4, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/7x7_s2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_1/3x3_b2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_2/3x3_b2')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_1/3x3_b3')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_2/3x3_b3')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_3/3x3_b3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)


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
    
    #x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='classifier')(x)

    #inputs = img_input
    # Create model.
    model = Model(inputs, x, name='PathNet_resnet50')

    return model

def get_Path_VGG19_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.vgg19.VGG19(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_CXN[0], TARGET_SIZE_CXN[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_VGG19_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.vgg19.VGG19(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_ResNet50_model_BL(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((INP_SIZE[0], INP_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    #bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # # General convolutional layer
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_ResNet50_model_512X(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[0])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    #x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    #x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    
    return model


def get_Path_ResNet50_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_ResNet50[0], TARGET_SIZE_ResNet50[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_actual_ResNet50_model_for_1024x(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((INP_SIZE[0], INP_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    #bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # # General convolutional layer
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_32x32x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ResNet50_model_for_2048x(INP_SIZE, num_feature_maps, num_classes):
    
   backbone = tf.keras.applications.resnet50.ResNet50(
       weights=None,
       include_top=False,
       classes=num_classes,
       input_shape=((TARGET_SIZE_ResNet50[0], TARGET_SIZE_ResNet50[1], num_feature_maps[-1])),
   )
   backbone.trainable = True

   #pdb.set_trace()
   bn_axis = 3
   inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
   x = layers.Rescaling(scale=1.0 / 255)(inputs)
   #x = learnable_resizer(x)
   
   # General convolutional layer
   x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
   x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
   x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
   x = Activation('relu')(x)
   x = MaxPooling2D((3, 3), strides=(2, 2))(x)
   
   # Block 2
   x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
   x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
   x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs21bn')(x)
   x = Activation('relu')(x)
   x = MaxPooling2D((3, 3), strides=(2, 2))(x)
   
   x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
   
   # After applying the resizer model, define feature extractor and classifier...
   x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
   x = tf.keras.layers.GlobalAveragePooling2D()(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
   model = tf.keras.Model(inputs, final_output)
   
    
   return model


def get_Path_ResNet50_model_AVGP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ResNet50_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_ResNet50_model_WOMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP_LCDB[0], TARGET_SIZE_WOMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<

    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ResNet50_model_WOMP_LCDB_512x(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<

    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_ResNet50_model_WMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WMP_LCDB[0], TARGET_SIZE_WMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ResNet50_ResizerAVP_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_ResNet50V2_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

        #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ResNet50V2_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model




def get_Path_ResNet152_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ResNet152(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ResNet152_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ResNet152(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_DenseNet121_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_DenseNet[0], TARGET_SIZE_DenseNet[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_DenseNet121_model_BL(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((INP_SIZE[0], INP_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # # General convolutional layer
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_DenseNet121_AVGP_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_DenseNet[0], TARGET_SIZE_DenseNet[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_DenseNet121_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
        

    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_DenseNet121_model_WOMP_1024X(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
        

    backbone.trainable = True

    #pdb.set_trace()
    
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_DenseNet121_model_WOMP_2048X(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
        

    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # Block 1 : General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 3
    x = Conv2D(num_feature_maps[2], (3, 3), activation='relu', padding='same', name='conv3/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[2], (3, 3), activation='relu', padding='same', name='conv3/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_DenseNet121_model_WMXP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WMXP[0], TARGET_SIZE_WMXP[1], num_feature_maps[-1])),
    )
    
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_DenseNet121_model_WOMP_LCDB_512x(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    
    #pdb.set_trace()
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_DenseNet121_model_WOMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP_LCDB[0], TARGET_SIZE_WOMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #pdb.set_trace()
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # General convolutional layer
    x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    #pdb.set_trace()
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_DenseNet121_model_WMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WMP_LCDB[0], TARGET_SIZE_WMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
        
    # General convolutional layer
    x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_EfficientNetB7_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    pdb.set_trace()
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_EfficientNetB7_model_BL(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((INP_SIZE[0], INP_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # # General convolutional layer
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    pdb.set_trace()
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_EfficientNetB5_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB5(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    #pdb.set_trace()
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_EfficientNetB7_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_EfficientNetB7_model_WOMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP_LCDB[0], TARGET_SIZE_WOMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    #pdb.set_trace()
        
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_EfficientNetB7_model_WOMP_LCDB_512x(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    #pdb.set_trace()
        
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_EfficientNetV2M_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.EfficientNetV2M(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_EfficientNetV2M_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.EfficientNetV2M(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_InceptionV3_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True


    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1/3x3_rs13/bn_1')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    
    pdb.set_trace()
        
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE BACKBONE NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_InceptionV3_model_BL(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((INP_SIZE[0], INP_SIZE[1], 3)),
    )
    backbone.trainable = True


    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # # General convolutional layer
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    # x = BatchNormalization(axis=bn_axis, name='conv1/3x3_rs13/bn_1')(x)
    # x = Activation('relu')(x)
    # #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # # Block 2
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    
    # pdb.set_trace()
        
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE BACKBONE NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_InceptionV3_model_final_test(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_CXN[0], TARGET_SIZE_CXN[1], num_feature_maps[-1])),
    )
    backbone.trainable = True


    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1/3x3_rs13/bn_1')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv3/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv3/3x3_rs21')(x)
    
    #pdb.set_trace()
        
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE BACKBONE NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model




def get_Path_InceptionV3_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_InceptionV3_model_WOMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP_LCDB[0], TARGET_SIZE_WOMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
        
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_InceptionV3_model_WOMP_LCDB_512x(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
        
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_InceptionV3_model_WMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WMP_LCDB[0], TARGET_SIZE_WMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_Xception_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Xception_model_BL(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((INP_SIZE[0], INP_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    # x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    # x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_Xception_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_Xception_model_WOMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP_LCDB[0], TARGET_SIZE_WOMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
        
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_Xception_model_WOMP_LCDB_512x(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
        
    # General convolutional layer
    #x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='FEXT_12x12x512')(x)
    #x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/FEXT_6x6x512')(x)
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


#$ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. A ConvNet for the 2020s ( CURRENTLY NOT AVAILABLE) <<<<<<<<<<<<<<<<<<<<<<<<<<<<

def get_Path_ConvNeXtBase_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.convnext.ConvNeXtBase(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_CXN[0], TARGET_SIZE_CXN[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs13')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<< 88,403,270 parameters
    
    # # Since the ConvNeXtBase model provide 32x32x1024 outputs from last layer... 
    # # Thus, we added some convolutional block after receiving the 
    # # After applying the resizer model, define feature extractor and classifier...
    
    # #  Block - 2 General convolutional layer
    # x = Conv2D(1024, (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11_n2')(x)
    # x = Conv2D(1024, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12_n2')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn_n2')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # #>>>>>>>>>>>. output size : 16x16x1024 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
    # # Block -1 
    # x = Conv2D(512, (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11_n1')(x)
    # x = Conv2D(512, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12_n1')(x)
    # x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn_n1')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # #>>>>>>>>>>>. output size : 8x8x1024 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
    
    pdb.set_trace()
    
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

    

def get_Path_ConvNeXtXLarge_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ConvNeXtXLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model
 
    
def get_Path_ConvNeXtXLarge_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ConvNeXtXLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

def get_Path_ConvNeXtXLarge_model_WOMP_LCDB(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.ConvNeXtXLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP_LCDB[0], TARGET_SIZE_WOMP_LCDB[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_NASNetLarge_model(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.NASNetLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv1/3x3_rs21')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_Path_NASNetLarge_model_WOMP(INP_SIZE, num_feature_maps, num_classes):
    
    backbone = tf.keras.applications.NASNetLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE_WOMP[0], TARGET_SIZE_WOMP[1], num_feature_maps[-1])),
    )
    backbone.trainable = True

    #pdb.set_trace()
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = learnable_resizer(x)
    
    # General convolutional layer
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, padding='same',name='conv1/3x3_rs11')(x)
    x = Conv2D(num_feature_maps[0], (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/3x3_rs12')(x)
    x = BatchNormalization(axis=bn_axis, name='cconv1/3x3_rs12/bn')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs121')(x)
    x = Conv2D(num_feature_maps[-1], (3, 3), activation='relu', padding='same', name='conv2/3x3_rs21')(x)
    #x = BatchNormalization(axis=bn_axis, name='cconv2/3x3_rs12/bn')(x)
    #x = Activation('relu')(x)
    
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # After applying the resizer model, define feature extractor and classifier...
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model

if __name__ == '__main__':
    
    
    input_shape = (1024,1024,3)
    
    #input_shape = (768,768,3)

    pooling='avg'
    classes=74
    num_feature_maps = [4,8]
    #include_top=True 
    #pdb.set_trace()
    #model = get_Path_DenseNet121_model_WOMP_LCDB(input_shape, num_feature_maps, classes)
    
    model =get_actual_ResNet50_model_for_1024x(input_shape, classes)
    #model = get_Path_EfficientNetB7_model_WOMP_LCDB(input_shape, num_feature_maps, classes)
    
    #model = get_Path_InceptionV3_model_WOMP_LCDB(input_shape, num_feature_maps, classes)
     
    model.summary(expand_nested=True) 
    
    
    #model.summary()
    

    
