#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:31:59 2023

@author: mza
"""
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pdb
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
    
#INP_SIZE = (768, 768)
TARGET_SIZE = (224, 224)
TARGET_Xnet_SIZE = (299, 299)

#INTERPOLATION = "bilinear"

INTERPOLATION = "bicubic"

def conv_block(x, filters, kernel_size, strides, activation=layers.LeakyReLU(0.2)):
    x = layers.Conv2D(filters, kernel_size, strides, padding="same", use_bias=False)(x)
    if activation:
        x = activation(x)
    x = layers.BatchNormalization()(x)
    return x


def res_block(x):
    inputs = x
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 3, 1, activation=None)
    return layers.Add()([inputs, x])

def get_learnable_resizer(inputs, 
    filters=16,
    num_res_blocks=1, 
    interpolation=INTERPOLATION):

    # We first need to resize to a fixed resolution to allow mini-batch learning
    naive_resize = layers.experimental.preprocessing.Resizing(*TARGET_SIZE,
            interpolation=interpolation)(inputs)

    # First conv block without Batch Norm
    x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Second conv block with Batch Norm
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # Intermediate resizing as bottleneck
    bottleneck = layers.experimental.preprocessing.Resizing(*TARGET_SIZE,
            interpolation=interpolation)(x)
    
    # Residual passes
    for _ in range(num_res_blocks):
        x = res_block(bottleneck)

    # Projection
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    x = layers.Add()([bottleneck, x])

    # Final resized image
    x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    final_resize = layers.Add()([naive_resize, x])

    return final_resize

def get_learnable_resizer_Xception(inputs, 
    filters=16,
    num_res_blocks=1, 
    interpolation=INTERPOLATION):

    # We first need to resize to a fixed resolution to allow mini-batch learning
    naive_resize = layers.experimental.preprocessing.Resizing(*TARGET_Xnet_SIZE,
            interpolation=interpolation)(inputs)

    # First conv block without Batch Norm
    x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Second conv block with Batch Norm
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # Intermediate resizing as bottleneck
    bottleneck = layers.experimental.preprocessing.Resizing(*TARGET_Xnet_SIZE,
            interpolation=interpolation)(x)
    
    # Residual passes
    for _ in range(num_res_blocks):
        x = res_block(bottleneck)

    # Projection
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    x = layers.Add()([bottleneck, x])

    # Final resized image
    x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    final_resize = layers.Add()([naive_resize, x])

    return final_resize

def get_model_LR_with_VGG19_with_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.vgg19.VGG19(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model

def get_model_VGG19_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.vgg19.VGG19(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    #x = get_learnable_resizer(x)
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


def get_model_LR_with_VGG19_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.vgg19.VGG19(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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


def get_model_LR_with_ResNet50(INP_SIZE, num_classes):
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


def get_model_LR_with_ResNet50_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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

def PathNet_RESNET50_FSRCT_with_LR(INP_SIZE, num_classes):
    
    bn_axis = 3
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    
    x = get_learnable_resizer(x)
    
    # # General convolutional layer
    # x = Conv2D(4, (3, 3), use_bias=False, strides=(2, 2), padding='same',name='conv1/7x7_s2')(x)
    # x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # # Block 2
    # x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_1/3x3_b2')(x)
    # x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2_2/3x3_b2')(x)
    # #x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # # Block 3
    # # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_1/3x3_b3')(x)
    # # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_2/3x3_b3')(x)
    # # x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3_3/3x3_b3')(x)
    # # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)


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



 #Identity Mappings in Deep Residual Networks (CVPR 2016)
def get_model_LR_with_ResNet50V2(INP_SIZE, num_classes):
    backbone = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_ResNet50V2_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    #pdb.set_trace()
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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


def get_model_LR_with_DenseNet121(INP_SIZE, num_classes):
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


def get_model_LR_with_DenseNet121_without_TOP(INP_SIZE,  num_classes):
    
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    #pdb.set_trace()
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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


def get_model_LR_with_EfficientNetB7(INP_SIZE, num_classes):
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_EfficientNetB7_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    
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

def get_model_LR_with_EfficientNetB6_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB6(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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


def get_model_LR_with_EfficientNetV2M_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.EfficientNetV2M(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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



def get_model_LR_with_InceptionV3(INP_SIZE, num_classes):
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_InceptionV3_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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


def get_model_LR_with_XceptionNet_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_Xnet_SIZE[0], TARGET_Xnet_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    #pdb.set_trace()
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer_Xception(x)
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



#$ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. A ConvNet for the 2020s ( CURRENTLY NOT AVAILABLE) <<<<<<<<<<<<<<<<<<<<<<<<<<<<

def get_model_LR_with_ConvNeXtBase_without_TOP(INP_SIZE, num_classes):
    
    # >>>>>>>>>>>>>>>>>>>>. For TF2.12 version.....
    #   https://www.tensorflow.org/api_docs/python/tf/keras/applications/convnext/ConvNeXtLarge
    
    #bn_axis = 3
     
    backbone = tf.keras.applications.convnext.ConvNeXtBase(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
   
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    x = backbone(x)  # >>>>>>>>>>>>>>>>>>>>>>>  CALL THE RESIZER NETWORK <<<<<<<<<<<<<<<<<<<<<<<<<88,404,045 parameters
    
    # After applying the resizer model, define feature extractor and classifier...
    # Since the ConvNeXtBase model provide 32x32x1024 outputs from last layer... 
    # Thus, we added some convolutional block after receiving the 
    # After applying the resizer model, define feature extractor and classifier...
    
        
    # pdb.set_trace()
    
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
    #>>>>>>>>>>>. output size : 8x8x1024 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
      
    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu', padding='same', name='FEXT_7x7x512')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',name='FEXL_512')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    final_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='final_output')(x)
    model = tf.keras.Model(inputs, final_output)
    
    return model


def get_model_LR_with_ConvNeXtXLarge_with_TOP(INP_SIZE, num_classes):
    backbone = tf.keras.applications.ConvNeXtXLarge(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_ConvNeXtXLarge_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.convnext.ConvNeXtXLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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

def get_model_LR_with_ConvNeXtLarge_without_TOP(INP_SIZE, num_classes):
    
    #   https://www.tensorflow.org/api_docs/python/tf/keras/applications/convnext/ConvNeXtLarge
    
    backbone = tf.keras.applications.convnext.ConvNeXtLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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





def get_model_LR_with_NASNet_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.NASNetLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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



#Revisiting ResNets: Improved Training and Scaling Strategie ( CURRENTLY NOT AVAILABLE)
def get_model_LR_with_ResNetRS152(INP_SIZE, num_classes):
    backbone = tf.keras.applications.ResNet152(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


def get_model_LR_with_ResNetRS152_without_TOP(INP_SIZE, num_classes):
    
    backbone = tf.keras.applications.ResNet152(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = get_learnable_resizer(x)
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


# How to call the model........

def main():
    #print("Hello World!")

    #learnable_resizer = get_learnable_resizer_v2()
    #learnable_resizer.summary()
    
    
    INP_SIZE = (1024, 1024)
    num_classes = 74
    
    #pdb.set_trace()
    
    #model = get_model_LR_with_ConvNeXtBase_without_TOP(INP_SIZE, num_classes)
    model =  PathNet_RESNET50_FSRCT_with_LR(INP_SIZE, num_classes)
    pdb.set_trace()
    
    #model.summary()
  
    # # To see the complete summary with backbone...
    model.summary(expand_nested=True) 
    
    # layer_dict = dict([(layer.name,layer) for layer in model.layers])
    # layer_name_7x7x512 = 'FEXT_7x7x512'
    # feature_extractor_2D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_7x7x512].output)
    # feature_extractor_2D.summary()
    
    
    # layer_name_512 = 'FEXL_512'
    # feature_extractor_1D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_512].output)
    # feature_extractor_1D.summary()

if __name__ == "__main__":
    main()