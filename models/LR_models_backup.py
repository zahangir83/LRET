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

INP_SIZE = (768, 768)
TARGET_SIZE = (224, 224)
TARGET_Xnet_SIZE = (299, 299)

INTERPOLATION = "bilinear"

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


# def get_learnable_resizer(filters=16, num_res_blocks=1, interpolation=INTERPOLATION):
#     inputs = layers.Input(shape=[None, None, 3])

#     # First, perform naive resizing.
#     naive_resize = layers.Resizing(
#         *TARGET_SIZE, interpolation=interpolation
#     )(inputs)

#     # First convolution block without batch normalization.
#     x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding="same")(inputs)
#     x = layers.LeakyReLU(0.2)(x)

#     # Second convolution block with batch normalization.
#     x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.BatchNormalization()(x)

#     # Intermediate resizing as a bottleneck.
#     bottleneck = layers.Resizing(
#         *TARGET_SIZE, interpolation=interpolation
#     )(x)

#     # Residual passes.
#     for _ in range(num_res_blocks):
#         x = res_block(bottleneck)

#     # Projection.
#     x = layers.Conv2D(
#         filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False
#     )(x)
#     x = layers.BatchNormalization()(x)

#     # Skip connection.
#     x = layers.Add()([bottleneck, x])

#     # Final resized image.
#     x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
#     final_resize = layers.Add()([naive_resize, x])

#     return tf.keras.Model(inputs, final_resize, name="learnable_resizer")

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


# def get_learnable_resizer_Xception(filters=16, num_res_blocks=1, interpolation=INTERPOLATION):
#     inputs = layers.Input(shape=[None, None, 3])

#     # First, perform naive resizing.
#     naive_resize = layers.Resizing(
#         *TARGET_Xnet_SIZE, interpolation=interpolation
#     )(inputs)

#     # First convolution block without batch normalization.
#     x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding="same")(inputs)
#     x = layers.LeakyReLU(0.2)(x)

#     # Second convolution block with batch normalization.
#     x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.BatchNormalization()(x)

#     # Intermediate resizing as a bottleneck.
#     bottleneck = layers.Resizing(
#         *TARGET_Xnet_SIZE, interpolation=interpolation
#     )(x)

#     # Residual passes.
#     for _ in range(num_res_blocks):
#         x = res_block(bottleneck)

#     # Projection.
#     x = layers.Conv2D(
#         filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False
#     )(x)
#     x = layers.BatchNormalization()(x)

#     # Skip connection.
#     x = layers.Add()([bottleneck, x])

#     # Final resized image.
#     x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
#     final_resize = layers.Add()([naive_resize, x])

#     return tf.keras.Model(inputs, final_resize, name="learnable_resizer")

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

def get_model_LR_with_VGG19_with_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.vgg19.VGG19(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model


def get_model_LR_with_VGG19_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
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
    x = learnable_resizer(x)
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


def get_model_LR_with_ResNet50(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


def get_model_LR_with_ResNet50_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
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
    x = learnable_resizer(x)
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



 #Identity Mappings in Deep Residual Networks (CVPR 2016)
def get_model_LR_with_ResNet50V2(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_ResNet50V2_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
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
    x = learnable_resizer(x)
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


def get_model_LR_with_DenseNet121(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.densenet.DenseNet121(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


def get_model_LR_with_DenseNet121_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
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
    x = learnable_resizer(x)
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


def get_model_LR_with_DenseNet121_without_TOP_V2(INP_SIZE, num_classes):
    
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
    x = get_learnable_resizer_v2(x)
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

def get_model_LR_with_EfficientNetB7(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_EfficientNetB7_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB7(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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

def get_model_LR_with_EfficientNetB6_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.efficientnet.EfficientNetB6(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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


def get_model_LR_with_EfficientNetV2M_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.EfficientNetV2M(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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



def get_model_LR_with_InceptionV3(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_InceptionV3_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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


def get_model_LR_with_XceptionNet_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.Xception(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_Xnet_SIZE[0], TARGET_Xnet_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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



#$ A ConvNet for the 2020s ( CURRENTLY NOT AVAILABLE)
def get_model_LR_with_ConvNeXtXLarge(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.ConvNeXtXLarge(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)

def get_model_LR_with_ConvNeXtXLarge_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.ConvNeXtXLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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

def get_model_LR_with_ConvNeXtLarge_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.ConvNeXtLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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


def get_model_LR_with_ConvNeXtBase_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.ConvNeXtBase(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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


def get_model_LR_with_NASNet_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.NASNetLarge(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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
def get_model_LR_with_ResNetRS152(INP_SIZE, learnable_resizer, num_classes):
    backbone = tf.keras.applications.ResNet152(
        weights=None,
        include_top=True,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


def get_model_LR_with_ResNetRS152_without_TOP(INP_SIZE, learnable_resizer, num_classes):
    
    backbone = tf.keras.applications.ResNet152(
        weights=None,
        include_top=False,
        classes=num_classes,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True
    
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
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
    print("Hello World!")

    #learnable_resizer = get_learnable_resizer_v2()
    #learnable_resizer.summary()
    
    
    INP_SIZE = (1024, 1024)
    num_classes = 2
    
    pdb.set_trace()
    
    model = get_model_LR_with_DenseNet121_without_TOP_V2(INP_SIZE, num_classes)
    # pdb.set_trace()
    
    model.summary()
  
    # # To see the complete summary with backbone...
    # model.summary(expand_nested=True) 
    
    # layer_dict = dict([(layer.name,layer) for layer in model.layers])
    # layer_name_7x7x512 = 'FEXT_7x7x512'
    # feature_extractor_2D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_7x7x512].output)
    # feature_extractor_2D.summary()
    
    
    # layer_name_512 = 'FEXL_512'
    # feature_extractor_1D = Model(inputs=model.inputs,outputs=layer_dict[layer_name_512].output)
    # feature_extractor_1D.summary()

if __name__ == "__main__":
    main()