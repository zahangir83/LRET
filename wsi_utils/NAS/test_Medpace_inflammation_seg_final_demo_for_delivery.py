import os,time,cv2 
import argparse
import numpy as np
from os.path import join as join_path
from utils import helpers as helpers
from wsi_utils import hpf_patches_utils_ts
from wsi_utils import svs_utils_final_one as svs_utils

kernel = np.ones((3,3), np.uint8) 
font = cv2.FONT_HERSHEY_SIMPLEX
from tensorflow import keras
import shutil
import pdb

import tensorflow as tf
smooth = 1.

# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

valid_images = ['.svs','.jpg','.png']
    
parser = argparse.ArgumentParser()
parser.add_argument('--project_name_segmentation', type=str, default="inflam_seg_R2U_Net_MP_and_PUB_DB_DC_final", help='Name of your project')
parser.add_argument('--HPF_height', type=int, default=1024, help='Height of cropped input image to network') #1152
parser.add_argument('--HPF_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--seg_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--seg_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--model_seg', type=str, default="R2UNet", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--input_svs_image_path', type=str, default="/home/mza/Desktop/MedPace_projects/steatosis_detection_project/delivered_projects/database/wsi_hpf_for_testing/wsi/single/", help='Dataset you are using.')
 
args = parser.parse_args()
print("creating all the necessary diretoreis:")
project_log_path_seg = join_path('experimental_logs/', args.project_name_segmentation+'/')   
training_log_saving_path_seg = join_path(project_log_path_seg,'training/')
testing_log_saving_path_seg_1 = join_path(project_log_path_seg,'testing/')
weight_loading_path_seg = join_path(project_log_path_seg,'weights/')
trained_model = join_path(project_log_path_seg,'trained_model/')

# create pathces from the WSI
HPFs_size = (args.HPF_height,args.HPF_width)  
# Extract specific size of the patches from WSI
patches_dir, wsi_filenames = svs_utils.extract_same_size_patches_from_wsi_final(args.input_svs_image_path, testing_log_saving_path_seg_1, HPFs_size)

#Read the extracted patches from the directory (if available)
#patches_dir = '/home/mza/Desktop/NASH_score_public_dataset/Ballooning_seg_testing_phase/experimental_logs/Seatosis_seg_R2UNet/testing/2MGY7D24/'
slide_name = patches_dir.split('/')[-2]
#pdb.set_trace()
if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_1,slide_name+'_outputs')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_1,slide_name+'_outputs'))        

testing_log_saving_path_seg_2 = join_path(testing_log_saving_path_seg_1,slide_name+'_outputs')

# saving all of the ouput patches...
if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_2,'output_patches/')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_2,'output_patches/'))   
testing_log_saving_path_seg_patcehs = join_path(testing_log_saving_path_seg_2,'output_patches/') 

if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_2,'output_images/')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_2,'output_images/'))   
testing_log_saving_path_seg_images = join_path(testing_log_saving_path_seg_2,'output_images/') 

if not os.path.isdir("%s/%s"%(testing_log_saving_path_seg_2,'FP_images/')):
    os.makedirs("%s/%s"%(testing_log_saving_path_seg_2,'FP_images/'))   
testing_log_saving_path_FP_images = join_path(testing_log_saving_path_seg_2,'FP_images/') 
# Find the json log fle for patches and copy to
json_file_name = [x for x in sorted(os.listdir(patches_dir)) if x[-5:] == '.json']
json_file_name_with_dir = join_path(patches_dir,json_file_name[0]) 
shutil.copy2(json_file_name_with_dir,testing_log_saving_path_seg_patcehs)
# Model input size for Segmentation model...
seg_net_input = (args.seg_height, args.seg_width,3)
num_classes = 2
print('-'*30)
print('Loading mean and std for the segmentation and detection models...')
print('-'*30)
# load mean and std for sample normalization for segmentation model
mean_path_grady = join_path(training_log_saving_path_seg,'inflammaton_seg_mean.npy')
std_path_grady = join_path(training_log_saving_path_seg,'inflammaton_seg_std.npy')
print('-'*30)
print('Loading weights for all model...')
print('-'*30)
model_seg_loading_path = join_path(trained_model,'inflammation_seg_model.h5')
model_segmentation = tf.keras.models.load_model(model_seg_loading_path,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})   
model_segmentation.summary()
# Load the testing data from data directory...
print("Loading image from the directorty ...")              
images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png' or '.svs' or '.tif']
print('Total number of images:'+str(len(images_name)))

img_name = images_name[0].split('.')[0]
img_ext = images_name[0].split('.')[1]

if img_ext.lower() not in valid_images:
    print('Valid input image')
else:
    print('Please input jpg,png or svs')
    
Total_steatosis_cell_wsi = 0
Total_pixels = 0
total_segmentated_pixels_wsi = 0

for i, img_name in enumerate(images_name):
    
    ext = os.path.splitext(img_name)[1]
    if ext == '.json':
        continue  
    
    image_id = os.path.splitext(img_name)[0]               
    img_name_wo_ext = os.path.splitext(img_name)[0]
    img_path = os.path.join(patches_dir,img_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    patch_h = args.seg_height
    patch_w = args.seg_width
    
    original_img = img   
    # Extract patches (256x256) for segmentation tasks...
    patch_h = args.seg_height
    patch_w = args.seg_width
    print(img_path)
    img_patches, img_patches_number, num_rows,num_columns = hpf_patches_utils_ts.extract_patches_from_image(img,patch_h,patch_w)
    mask_patches_number  = img_patches_number
    
    ac_x_test = img_patches
    x_test = ac_x_test
    # Normalize the input images with respect to the existing mean and std for segmentation task..
    mean_grady = np.load(mean_path_grady)
    std_grady = np.load(std_path_grady)
    x_test_seg = x_test.astype('float32')
    x_test_seg -= mean_grady
    x_test_seg /= std_grady 
    x_test_seg = x_test_seg.reshape(x_test_seg.shape[0], args.seg_height, args.seg_width,3)
    # Apply segmentation model...
    t1 = time.time()
    y_hat_seg = model_segmentation.predict(x_test_seg)
    t2=time.time()                   
    print ("Total time for segmentation:")
    t_f = t2-t1
    print("Total time:",t_f)                   
    number_test_samples = x_test_seg.shape
    y_hat_seg = np.squeeze(y_hat_seg)
    # Apply thresholding...and reconstruct original image from the patches
    pred_masks_seg = 255.0*(y_hat_seg[:,:,:] >= 0.5)               
    reconstructed_image = hpf_patches_utils_ts.image_from_patches(ac_x_test, num_rows,num_columns)
    reconstructed_pred_seg = hpf_patches_utils_ts.image_from_patches(pred_masks_seg, num_rows,num_columns)
    # Performing morphological operations.....
    reconstructed_pred_seg_morph = helpers.perform_morphological_operations(reconstructed_pred_seg)            
    reconstructed_pred_seg_morph_final = 255.0*(reconstructed_pred_seg_morph[:,:] >= 0.5) 
    mask_gray_img = reconstructed_pred_seg_morph

    ac_img_name =str(image_id)+'_actual_img.jpg'
    y_pred_name_seg = str(image_id)+'_image_pred_seg'+'.jpg'  
    y_pred_name_combind = str(image_id)+'_image_pred_seg_pp.jpg' 
                                           
    final_des_img = os.path.join(testing_log_saving_path_seg_patcehs,ac_img_name)
    final_des_pred_seg = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_seg)
    final_des_pred_combind = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_combind)
                                    
    cv2.imwrite(final_des_img,original_img)
    cv2.imwrite(final_des_pred_seg,reconstructed_pred_seg)
    cv2.imwrite(final_des_pred_combind,reconstructed_pred_seg_morph_final)
    

svs_utils.patches_to_binary_image_from_seg_class_inflamation_masks_medpace_final(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)

## Merging all output patches together...
#svs_utils.patches_to_actual_image_and_ROI_refined_mask_medpace(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
#svs_utils.patches_to_binary_image_from_seg_class_steatosis_masks_medpace_final(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
## create heatmaps outputs from roi from seg+class masks
#svs_utils.create_heatmaps_from_roi_images_from_class_plus_seg_mask_st(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)

