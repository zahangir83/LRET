from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os,sys,json
import subprocess
from os.path import join as join_path
import pdb
# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
#from builders import model_builder

from  models.R2UNet import build_R2UNetE,build_R2UNetED,build_R2UNetED_final,build_R2UNetED_DP_final,build_DeltaNetAB_final,Dilated_R2UNetED_DP_final

from utils import dataset_utils
#import dataset_utils as data_utils
import matplotlib.pyplot as plt

abspath = os.path.dirname(os.path.abspath(__file__))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="inflam_seg_R2U_Net_MP_and_PUB_DB_DC_final", help='Name of your project')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')

parser.add_argument('--dataset_path', type=str, default="/home/mza/Desktop/NASH_score_public_dataset/inflammation_segmentation_project/data/MP_and_PUB_segDB/training/", help='Dataset you are using.')
#parser.add_argument('--validation_data_path', type=str, default="/home/deeplens/zahangir/cvpr_mitosis_detection/mitosis_detection_phase/database/Images/images_masks/validation_images_masks/Group_1/F0003_patches_images_masks/", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
#parser.add_argument('--model', type=str, default="ResidualUNet", help='The model you are using. See model_builder.py for supported models')
#parser.add_argument('--model', type=str, default="R2UNet", help='The model you are using. See model_builder.py for supported models')

parser.add_argument('--model', type=str, default="R2UNet_DP", help='The model you are using. See model_builder.py for supported models')
#parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()

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
weight_saving_path = join_path(project_path,'weights/')
model_saving_path = join_path(project_path,'trained_model/')

#pdb.set_trace()
# Load the data
print("Loading the data ...")
ac_x_data,gray_x_data,y_data = dataset_utils.read_images_and_masks_for_inflammation_seg(args.dataset_path,args.crop_height, args.crop_width)
x_data = ac_x_data
x_data,y_data,mean,std = dataset_utils.samples_normalization (x_data, y_data)



mean_saving_dir = training_log_saving_path+'nuclei_mean.npy'
std_saving_dir = training_log_saving_path+'nuclei_std.npy'

np.save(mean_saving_dir,mean)
np.save(std_saving_dir,std)

# reshape samples to feed to the model..
x_data = x_data.reshape(x_data.shape[0], args.crop_height, args.crop_width,3)
y_data = y_data.reshape(y_data.shape[0], args.crop_height, args.crop_width,1)
ac_x_train,x_train,y_train,ac_x_val,x_val,y_val = dataset_utils.split_data_train_val (ac_x_data,x_data,y_data)

ac_x_val_path = join_path(training_log_saving_path+'ac_x_val.npy')
x_val_path = join_path(training_log_saving_path+'x_val.npy')
y_val_path = join_path(training_log_saving_path+'y_val.npy')

np.save(ac_x_val_path,ac_x_val)
np.save(x_val_path,x_val)
np.save(y_val_path,y_val)
# Model input size:
net_input = (args.crop_height, args.crop_width,3)
num_classes = 2

# load model here..
#pdb.set_trace()
print("Model building...")
#model, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)


model = build_R2UNetED_DP_final(net_input,num_classes)  

model.summary()

#

pdb.set_trace()

print("\n ***** Training details *****")
print("Dataset -->", args.dataset_path)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)


print("\n ** Training model ...")

history = model.fit(x_train,y_train,batch_size=args.batch_size,epochs=args.num_epochs,validation_data=(x_val,y_val))

#pdb.set_trace()
model_saving_path = model_saving_path+'model.h5'
model.save(model_saving_path)  

# save the training log as json file..
training_log = {}
training_log["Model_loss"] = history.history['loss']
training_log["Accuracy"] = history.history['acc']
training_log["MSE"] = history.history['mse']
training_log["val_loss"] = history.history['val_loss']
training_log["val_Accuracy"] = history.history['val_acc']
training_log["val_MSE"] = history.history['val_mse']

# plot and save the log file 
plots_saving_path = os.path.join(training_log_saving_path,'Training_acc.png')
model_acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
plt.plot(model_acc, color="tomato", linewidth=2)
plt.plot(val_acc, color="steelblue", linewidth=2)  
plt.legend(["Training","Validation"],loc=4)
plt.title("Training and validation accuracy.")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")   
plt.grid()
plt.savefig(plots_saving_path)
plt.show() 
plt.clf()
plt.close()


# make experimental log saving path...
json_file = os.path.join(training_log_saving_path,'model_training_log.json')
with open(json_file, 'w') as file_path:
    json.dump(training_log, file_path, indent=4, sort_keys=True)

#pdb.set_trace()

print("Testing model..")
# Predict on test data
y_val_hat = model.predict(x_val,verbose=1)


img_idx = 1
fig, ax = plt.subplots(2,5,figsize=(12,6)) 
num_img =5    

for img_idx in range(num_img):
    plt.axis('off')
    ax[0][img_idx].imshow(x_val[img_idx,:,:,0])
    ax[1][img_idx].imshow(y_val_hat[img_idx,:,:,0], cmap='gray')
   
fig.savefig(testing_log_saving_path+"Output_"+str(img_idx)+".png")



