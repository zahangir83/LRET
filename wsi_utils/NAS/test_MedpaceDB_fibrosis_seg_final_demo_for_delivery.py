import os,time,cv2 
import argparse
import numpy as np
from os.path import join as join_path
from utils import helpers as helpers
from wsi_utils import hpf_patches_utils_ts
from wsi_utils import svs_utils_final_one as svs_utils
from models import R2UNet as seg_models
import tensorflow as tf
kernel = np.ones((3,3), np.uint8) 
font = cv2.FONT_HERSHEY_SIMPLEX

from tensorflow import keras
import shutil
import pdb

valid_images = ['.svs','.jpg','.png']

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
    
parser = argparse.ArgumentParser()
parser.add_argument('--project_name_segmentation', type=str, default="fibrosis_seg_R2U_Net_DC_MDB_PIL", help='Name of your project')
parser.add_argument('--project_name_classification', type=str, default="project_steatosis_ResNet50_BBBL", help='Name of your project')
parser.add_argument('--HPF_height', type=int, default=1024, help='Height of cropped input image to network') #1152
parser.add_argument('--HPF_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--seg_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--seg_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--clas_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--clas_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--model_seg', type=str, default="R2UNet", help='The model you are using. See model_builder.py for supported models')
#parser.add_argument('--input_svs_image_path', type=str, default="/home/mza/Desktop/MedPace_projects/steatosis_detection_project/delivered_projects/database/wsi_hpf_for_testing/wsi/single/", help='Dataset you are using.')
parser.add_argument('--input_svs_image_path', type=str, default="/media/mza/Data-Storage/MedPace_projects/MedPace_new_samples/TRI/one/", help='Dataset you are using.')

args = parser.parse_args()

print("creating all the necessary diretoreis:")
project_log_path_seg = join_path('experimental_logs/', args.project_name_segmentation+'/')   
training_log_saving_path_seg = join_path(project_log_path_seg,'training/')
testing_log_saving_path_seg_1 = join_path(project_log_path_seg,'testing/')
weight_loading_path_seg = join_path(project_log_path_seg,'weights/')
trained_model = join_path(project_log_path_seg,'trained_model/')

# create pathces from the WSI
HPFs_size = (args.HPF_height,args.HPF_width)  
# Extract specific size of the patches...
#pdb.set_trace()

patches_dir, wsi_filenames = svs_utils.extract_same_size_patches_from_wsi_final(args.input_svs_image_path, testing_log_saving_path_seg_1, HPFs_size)

#patches_dir ='/home/mza/Desktop/NASH_score_public_dataset/Fibrosis_seg_testing_phase/experimental_logs/fibrosis_seg_R2U_Net_DC_MDB_PIL/testing/101-048_LBIO1_TRI/'

# Model input size for Segmentation model...
seg_net_input = (args.seg_height, args.seg_width,3)
num_classes = 2
print("Model building...")
print('-'*30)
print('Loading mean and std for the segmentation and detection models...')
print('-'*30)
# load mean and std for sample normalization for segmentation model
mean_path_grady = join_path(training_log_saving_path_seg,'fibrosis_seg_mean.npy')
std_path_grady = join_path(training_log_saving_path_seg,'fibrosis_seg_std.npy')
# load weights to model
print('-'*30)
print('Loading weights for all model...')
print('-'*30)
model_path = join_path(trained_model,'fibrosis_seg_model.h5')
model_segmentation = tf.keras.models.load_model(model_path,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})   
print("Predicting")
model_segmentation.summary()

for wsi_filename in wsi_filenames:

    slide_name = os.path.splitext(wsi_filename)[0]

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


    # Load the testing data from data directory...
    print("Loading image from the directorty ...")              
    images_name = [x for x in sorted(os.listdir(patches_dir)) if x[-4:] == '.jpg' or '.png' or '.svs' or '.tif']
    print('Total number of images:'+str(len(images_name)))
    Total_steatosis_cell_wsi = 0
    Total_pixels = 0
    total_segmentated_pixels_wsi = 0

    for i, img_name in enumerate(images_name):
        
        ext = os.path.splitext(img_name)[1]    
        img_name_wo_ext = os.path.splitext(img_name)[0]
        img_path = os.path.join(patches_dir,img_name)
        if ext == '.json':
            continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        patch_h = args.seg_height
        patch_w = args.seg_width
        
        original_img = img
        # Extract patches (256x256) for segmentation and detection...
        patch_h = args.seg_height
        patch_w = args.seg_width
        img_patches, img_patches_number, num_rows,num_columns =hpf_patches_utils_ts.extract_patches_from_image(img,patch_h,patch_w)
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
        # Apply thresholding...and # Reconstruct original image from the patches
        pred_masks_seg = 255.0*(y_hat_seg[:,:,:] >= 0.8)               
        reconstructed_image = hpf_patches_utils_ts.image_from_patches(ac_x_test, img_patches_number, num_rows,num_columns)
        reconstructed_pred_seg = hpf_patches_utils_ts.image_from_patches(pred_masks_seg, mask_patches_number, num_rows,num_columns)
        # Perform morphological operations.....
        reconstructed_pred_seg_morph = helpers.perform_morphological_operations(reconstructed_pred_seg)            
        reconstructed_pred_seg_morph_final = 255.0*(reconstructed_pred_seg_morph[:,:] >= 0.5) 
        #final_mask = reconstructed_pred_seg_morph 
        generated_mask_seg_clas = reconstructed_pred_seg_morph
        uniques_values = np.unique(generated_mask_seg_clas)
        print(uniques_values)         

        # calculate the steatosis_pixel in HPF and accumulate for WSI
        num_pxls_steatosis_cells_HPFs = np.sum((generated_mask_seg_clas)==255)
        print('Num_pxls_steatosis_cells_HPFs  :'+str(num_pxls_steatosis_cells_HPFs))
        Total_steatosis_cell_wsi = Total_steatosis_cell_wsi+num_pxls_steatosis_cells_HPFs

        # calculate the total number of for HPF and accumulate for generating WSI   
        pixel_HPF_total = args.HPF_height*args.HPF_width
        Total_pixels = Total_pixels + pixel_HPF_total
        
        # Calculate the overall percentage of steatosis in HPF.....
        steatosis_per_hpf = (num_pxls_steatosis_cells_HPFs /pixel_HPF_total)*100
    
        # Saving the log in text file....
        log_file_name = img_name_wo_ext+'_pixels_fibrosis_seg_HPFs_logs.txt'
        patch_output_logs = join_path(testing_log_saving_path_seg_patcehs,log_file_name)        
        patch_logs = open(patch_output_logs, 'w')

        patch_logs.write("Total_pixels in HPF: "+str(pixel_HPF_total)
                            + "\n Total steatosis cells in HPF : " +str(num_pxls_steatosis_cells_HPFs)
                            + "\n Steatosis pixels in percentage:"+str(steatosis_per_hpf)
                            #+ "\n Steatosis class called :"+str(class_indexes_all)
                            )                
                                    
        print('Processing for:'+str(img_name_wo_ext))           
        # saving output images...
        ac_img_name =img_name_wo_ext+'_act_img.jpg'
        y_pred_name_seg = img_name_wo_ext+'_wop_seg.jpg'  
        y_pred_name_seg_morph = img_name_wo_ext+'_wpp_seg.jpg'    
        
        final_des_img = os.path.join(testing_log_saving_path_seg_patcehs,ac_img_name)
        final_des_pred_seg = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_seg)
        final_des_pred_seg_morph = os.path.join(testing_log_saving_path_seg_patcehs,y_pred_name_seg_morph)

        cv2.imwrite(final_des_img,original_img)
        cv2.imwrite(final_des_pred_seg,reconstructed_pred_seg)
        cv2.imwrite(final_des_pred_seg_morph,generated_mask_seg_clas)


    json_filename_token = '.json'
    des_filename_token = '_wpp_seg.jpg'
    tissue_string = 'fibrosis'

    # Merging all output patches together...
    # svs_utils.patches_to_actual_image_and_ROI_refined_mask_medpace_fibrosis(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
    # svs_utils.patches_to_binary_mask_from_seg_fibrosis_medpace_final(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images)
    svs_utils.patches_to_binary_image_from_seg_class(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images, tissue_string, json_filename_token, des_filename_token)
    svs_utils.patches_to_binary_image_from_seg_class(testing_log_saving_path_seg_patcehs,testing_log_saving_path_seg_images, tissue_string, json_filename_token, '_wop_seg.jpg')
    # create heatmaps outputs from roi from seg+class masks
    #svs_utils.create_heatmaps_from_roi_images_from_class_plus_seg_mask_st(testing_log_saving_path_seg_images,testing_log_saving_path_seg_images)


