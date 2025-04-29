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
from models import PathNet_TF_V1
from tensorflow.keras.models import Model
import time
from train_utils import train_helper #import get_class_weights

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   #fname='flower_photos',
                                   #untar=True)
import tensorflow_datasets as tfds
from train_utils.train_helper import get_class_weights
#from tensorflow.keras.utils import multi_gpu_model
import json                                   
print(tf.__version__)
INTERPOLATION = "bilinear"
AUTO = tf.data.AUTOTUNE
#EPOCHS = 150
abspath = os.path.dirname(os.path.abspath(__file__))

try: 
    tpu = None
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: 
    strategy = tf.distribute.MirroredStrategy() 

print("Number of accelerators: ", strategy.num_replicas_in_sync)

def preprocess_dataset(image, TARGET_SIZE, label):
    image = tf.image.resize(image, (TARGET_SIZE[0], TARGET_SIZE[1]))
    label = tf.one_hot(label, depth=2)
    return (image, label)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="74CLS_Path_ENetV2_4A100GPUs_final", help='Name of your project')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset_path_train', type=str, default="/scratch_space/malom/project_74_classes/db_final_20K_final/train/", help='Dataset you are using.')
parser.add_argument('--input_height', type=int, default=1024, help='Height of cropped input image to network')
parser.add_argument('--input_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--target_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--target_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--num_class', type=int, default=74, help='Number of images in each batch')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of images in each batch')

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


#training directory
#data_dir_train = "/home/mza/Desktop/zahangir/SJCRH/OOD_Learning/Histology_analysis/dataset/Colon/train/"
data_dir_train = pathlib.Path(args.dataset_path_train)

#validation directory
# =============================================================================
#data_dir_valid = "/home/mza/Desktop/zahangir/SJCRH/OOD_Learning/Histology_analysis/dataset/Colon/val/"
#data_dir_valid = pathlib.Path(args.dataset_path_val)
# =============================================================================

#Split from same directory for validation
data_dir_valid=data_dir_train
print(data_dir_train)
print(data_dir_valid)
image_count = len(list(data_dir_train.glob('*/*.png')))
print("Training images" +str(image_count))
image_count = len(list(data_dir_valid.glob('*/*.png')))
print("Validation images" +str(image_count))


# It's good practice to use a validation split when developing your model. You will use 80% of the images for training and 20% for validation.
# In[ ]:
seed_train_val = 123  # must be the same ....
train_val_ratio = 0.2
shuffle_flag = True

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  validation_split=train_val_ratio,
  subset="training",
  label_mode='categorical',
  seed=seed_train_val,
  image_size=(INP_SIZE[0], INP_SIZE[1]),
  batch_size=args.batch_size)


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# In[74]:
print(train_ds)
#In[75]:
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  validation_split=train_val_ratio,
  subset="validation",
  label_mode='categorical',
  seed=seed_train_val,
  image_size=(INP_SIZE[0], INP_SIZE[1]),
  batch_size=args.batch_size)
#You can find the class names in the `class_names` attribute on these datasets.

# In[74]

# In[76]:
class_names = train_ds.class_names
print(class_names)

# In[79]:
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE
X_train = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
X_val = val_ds.cache().prefetch(1)

print("Configuring the dataset for better performance")
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print(train_ds)

### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  CALL THE MODEL   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

num_feature_maps = [3,3]

with strategy.scope():
    
    model = PathNet_TF_V1.get_Path_EfficientNetV2M_model(INP_SIZE, num_feature_maps, num_classes)

    model.compile(
          optimizer='adam',
          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])
    
    #model.compile(optimizer=tf.optimizers.sgd(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
    #                       metrics=['accuracy'])

#pdb.set_trace()

model.summary(expand_nested=True) 
 
checkpoint_filepath = os.path.join(traned_model_saving_path,args.project_name+'_'+str(INP_SIZE[0])+'x'+str(INP_SIZE[0])+'_best_model'+'.h5')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    #save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


class_weights_input = train_helper.get_class_weights(data_dir_train)

tm_start = time.time()
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS, 
  class_weight=class_weights_input,  # weighted train to better handle any class imbalance
  callbacks=[model_checkpoint_callback]
)
tm_end = time.time()
tt_time = tm_end - tm_start
# Note: You will only train for a few epochs so this tutorial runs quickly. 
# In[85]:
final_model_sv_path = os.path.join(traned_model_saving_path,'final_model.h5')
model.save(final_model_sv_path)
#pdb.set_trace()

# Note: You will only train for a few epochs so this tutorial runs quickly. 
# In[85]:
    
def plot_save_training_logs(history,tt_time, num_epchs, training_log_saving_path):
    
    training_log = {}
    training_log["model_loss"] = history.history['loss']
    training_log["accuracy"] = history.history['accuracy']
    training_log["val_loss"] = history.history['val_loss']
    training_log["val_Accuracy"] = history.history['val_accuracy']
    training_log["total_training_time"] = tt_time
    training_log["num_epoch"] = num_epchs

    #pdb.set_trace()
    # make experimental log saving path...
    json_file = os.path.join(training_log_saving_path,args.project_name+str(INP_SIZE[0])+'x'+str(INP_SIZE[0])+'_train_log.json')
    with open(json_file, 'w') as file_path:
        json.dump(str(training_log), file_path, indent=4, sort_keys=True)
     
    plots_saving_path = os.path.join(training_log_saving_path,'Training_acc_'+str(INP_SIZE[0])+'x'+str(INP_SIZE[0])+'.png')
    model_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(model_acc, color="tomato", linewidth=2)
    plt.plot(val_acc, color="steelblue", linewidth=2)  
    plt.legend(["Training","Validation"],loc=4)
    plt.title("Training and validation accuracy.")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")   
    plt.grid()
    plt.savefig(plots_saving_path)


# In[86]:

plot_save_training_logs(history,tt_time, EPOCHS, testing_log_saving_path)
