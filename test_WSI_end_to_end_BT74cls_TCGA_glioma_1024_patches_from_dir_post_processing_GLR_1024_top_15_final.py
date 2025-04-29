#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:47:58 2024

@author: malom
"""

import numpy as np
import pandas as pd
from os.path import join as join_path
import pdb
import os

# Path for 415 cases .......
SA_dataset_path = '/Users/malom/Desktop/zahangir/projects/computational_pathology/Large_scale_histopathology_analysis/database/74_cls_DB/mahmood_DB_Labels/TCAGA_glioma_survival_labels_data/'
output_path_from_model = '/Users/malom/Desktop/zahangir/projects/computational_pathology/Large_scale_histopathology_analysis/experimental_logs/74CLS_LR_ResNet50_4A100GPUs_final_101723/testing/TCGA_glioma_1024px_patches_GLR1024_ouptuts/'
output_path = '/Users/malom/Desktop/zahangir/projects/computational_pathology/Large_scale_histopathology_analysis/experimental_logs/74CLS_LR_ResNet50_4A100GPUs_final_101723/testing/TCGA_glioma_1024px_patches_GLR1024_ouptuts_merged/'


###### >>>>>>>>>>>>>>>>LOAD the Final Survival and Labels datasets... <<<<<<<<<<<<<<<<<<<<<<
survival_logs_with_labels415 = np.load(join_path(SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.npy'),allow_pickle=True)
pd_survival_logs_with_labels415 = pd.read_csv(join_path(SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.csv'))
directory_case_ids = pd_survival_logs_with_labels415['patient_ID']

# Encoding and decoding labels file........

end_decod_file_path = '/Users/malom/Desktop/zahangir/projects/computational_pathology/Large_scale_histopathology_analysis/database/74_cls_DB/Class_group_lineage/'

end_decod_file = np.load(os.path.join(end_decod_file_path,'Copy_of_GT_EC_GT_DEC_Group_Lineage_tvsnt_final_bao_revised.npy'),allow_pickle=True)
y_encoded_values = end_decod_file[:,0]
y_decoded_labels = end_decod_file[:,1]
#patient_ids_all = survival_logs_with_labels415[:,2]

concat_mat_SA_all = []
concat_mat_pred_all = []
concat_top_5_outputs_all = []

pdb.set_trace()

dlfv_rep_all = []
kkk = 0

for case_ID in directory_case_ids:
    print('Working for :', case_ID)
    #slide_name = os.path.splitext(wsi_filename)[0]
    patche_output_dir = os.path.join(output_path_from_model, case_ID+'_outputs')
    
    numpy_files = [x for x in sorted(os.listdir(os.path.join(patche_output_dir))) if x[-12:] == '_IDs_all.npy']
    name = numpy_files[0] 
    patch_IDs = name.split('_IDs')[0]

    
    ids_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_IDs_all.npy')
    ids = np.load(ids_path,allow_pickle='True')
    exd_ids = np.expand_dims(ids, axis=-1)
    
    class_call_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_names_all.npy')
    classes_predicted = np.load(class_call_path,allow_pickle='True')
    exd_classes_predicted = np.expand_dims(classes_predicted, axis=-1)
    
    lineage_call_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_lineage_all.npy')
    lineage_predicted = np.load(lineage_call_path,allow_pickle='True')
    exd_lineage_predicted = np.expand_dims(lineage_predicted, axis=-1)

    
    group_call_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_lineage_all.npy')
    group_predicted = np.load(group_call_path,allow_pickle='True')
    exd_group_predicted = np.expand_dims(group_predicted, axis=-1)
    
    tvstn_call_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_tumor_vs_not_tumor_all.npy')
    tvsnt_predicted = np.load(tvstn_call_path,allow_pickle='True')
    exd_tvsnt_predicted = np.expand_dims(tvsnt_predicted, axis=-1)
    
    conf_values_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_confvs_all.npy')
    conf_val_predicted = np.load(conf_values_path,allow_pickle='True')
    conf_val_predicted_max = conf_val_predicted.max(axis=1)
    exd_conf_val_predicted_max = np.expand_dims(conf_val_predicted_max, axis=-1)
    
    # Select top -5 call
    
    # max_index_col = np.argmax(conf_val_predicted, axis=1)
    # conf_val_predicted[int(max_index_col[0])] = 0.00
    
    # max_index_col_second = np.argmax(conf_val_predicted, axis=1)
    # result_idx_second_call = np.where(y_encoded_values == max_index_col_second[0])
    # second_class_call = y_decoded_labels[int(result_idx_second_call[0])]
    # conf_val_predicted[int(max_index_col_second[0])] = 0.00
    
    #pdb.set_trace()
    
    top_N = 15
    max_index_for_top_5 = np.argsort(-conf_val_predicted)
    top_5_index = max_index_for_top_5[:,:top_N]
    
    # Matched with the encoded vlaues and get the final index.....
    top_5_index_final = []
    for ii in range(top_5_index.shape[0]):
        
        dec_top_5_indv_row = []
        for kk in range(len(top_5_index[ii])):
            
            result_idx_second_call = np.where(y_encoded_values == top_5_index[ii, kk])
            dec_top_5_indv_row.append(y_decoded_labels[int(result_idx_second_call[0])])
        
        dec_top_5_indv_row_exd = np.expand_dims(dec_top_5_indv_row, axis = 0)
        top_5_index_final.append(dec_top_5_indv_row_exd)
    
    top_5_index_final = np.array(top_5_index_final)
    
    #top_5_index_final = np.squeeze(top_5_index_final)
    
    if top_5_index_final.shape[0]>1:
        top_5_index_final = np.squeeze(top_5_index_final)
    else:
        top_5_index_final = np.squeeze(top_5_index_final)
        top_5_index_final = np.expand_dims(top_5_index_final, axis=0)
    
    
    #concat_top_5_outputs_all.append(top_5_index_final)
    
    if kkk ==0:
        concat_top_5_outputs_all = top_5_index_final
    else:
        concat_top_5_outputs_all = np.concatenate((concat_top_5_outputs_all,top_5_index_final), axis =0)
        
    #pdb.set_trace()
    
    # create the prediction matrix ....
    
    concat_mat_prediction_logs = np.concatenate((exd_ids,exd_classes_predicted,exd_lineage_predicted,exd_group_predicted,exd_tvsnt_predicted,exd_conf_val_predicted_max),axis=1)
    concat_mat_prediction_logs = np.array(concat_mat_prediction_logs)
    
    # Loading the DL featues .........
    dlfv_path = os.path.join(os.path.join(patche_output_dir),patch_IDs+'_fvs_all.npy')
    dlfv_rep = np.load(dlfv_path,allow_pickle='True')

    #concat_mat_pred_all.append(concat_mat_prediction_logs)
    
    if kkk ==0:
        concat_mat_pred_all = concat_mat_prediction_logs
        dlfv_rep_all = dlfv_rep
    else:
        concat_mat_pred_all = np.concatenate((concat_mat_pred_all,concat_mat_prediction_logs), axis =0)
        dlfv_rep_all = np.concatenate((dlfv_rep_all,dlfv_rep), axis =0)

        
    # Process the data from the log file ....
    
    idx_case_ID = np.where(directory_case_ids == case_ID)
    print('Patient index :', idx_case_ID[0])
    
    logs_from_SA = survival_logs_with_labels415[idx_case_ID[0],:]
    
    logs_same_num_as_predicted_SA = []
    
    if len(ids)>1:
        
        for kk in range(len(ids)):
            
            logs_same_num_as_predicted_SA.append(logs_from_SA)
        
        logs_same_num_as_predicted_SA = np.squeeze(logs_same_num_as_predicted_SA)
    else:
        logs_same_num_as_predicted_SA = logs_from_SA
            
    logs_same_num_as_predicted_SA = np.array(logs_same_num_as_predicted_SA)
    
    # Merge the SA logs and prediction matrix....
    
    if kkk ==0:
        concat_mat_SA_all = logs_same_num_as_predicted_SA
    else:
        concat_mat_SA_all = np.concatenate((concat_mat_SA_all,logs_same_num_as_predicted_SA), axis =0)
    #concat_mat_pred_SA = np.concatenate((logs_same_num_as_predicted_SA,concat_mat_prediction_logs),axis=1)
    #concat_mat_pred_SA = np.array(concat_mat_pred_SA)
    
    #concat_mat_SA_all.append(logs_same_num_as_predicted_SA)

    kkk = kkk + 1
    #concat_mat_pred_SA_all.append(concat_mat_pred_SA)

#pdb.set_trace()


concat_mat_pred_all = np.array(concat_mat_pred_all)
concat_top_5_outputs_all = np.array(concat_top_5_outputs_all)
concat_mat_SA_all = np.array(concat_mat_SA_all)
dlfv_rep_all = np.array(dlfv_rep_all)

assert concat_mat_pred_all.shape[0]==concat_mat_SA_all.shape[0]==dlfv_rep_all.shape[0] == concat_top_5_outputs_all.shape[0]


df_concat_mat_pred_all =pd.DataFrame(concat_mat_pred_all)
df_concat_mat_pred_all.columns = ['IDs_pred','Class_pred','Lineage_pred','Group_pred','Tvsnt_pred','Conf_max_pred']
    

column_name_SA = pd_survival_logs_with_labels415.columns
df_concat_mat_SA_all = pd.DataFrame(concat_mat_SA_all)
column_name_SA = pd_survival_logs_with_labels415.columns
df_concat_mat_SA_all.columns = column_name_SA[1:]

df_merge_SA_pred_logs= pd.concat([df_concat_mat_SA_all, df_concat_mat_pred_all], axis=1)
outputs_saving_path_csv = os.path.join(output_path,'merge_SA_prediction_Logs_top_1_accuracy.csv')
outputs_saving_path_npy = os.path.join(output_path,'merge_SA_prediction_Logs_top_1_accuracy.npy')
df_merge_SA_pred_logs.to_csv(outputs_saving_path_csv)
np_df_merge_SA_pred_logs = df_merge_SA_pred_logs.to_numpy()
np.save(outputs_saving_path_npy,np_df_merge_SA_pred_logs)

# Save the Top-5 accuracy ....

#pdb.set_trace()
df_concat_top_5_outputs_all =pd.DataFrame(concat_top_5_outputs_all)
df_concat_top_5_outputs_all.columns = ['Class_pred_TOP_1','Class_pred_TOP_2','Class_pred_TOP_3','Class_pred_TOP_4','Class_pred_TOP_5','Class_pred_TOP_6','Class_pred_TOP_7','Class_pred_TOP_8','Class_pred_TOP_9','Class_pred_TOP_10','Class_pred_TOP_11','Class_pred_TOP_12','Class_pred_TOP_13','Class_pred_TOP_14','Class_pred_TOP_15']


top_5_accuracy_saving_path_csv = os.path.join(output_path,'top_5_accuracy.csv')
top_5_accuracy_saving_path_npy = os.path.join(output_path,'top_5_accuracy.npy')
df_concat_top_5_outputs_all.to_csv(top_5_accuracy_saving_path_csv)

np_df_concat_top_5_outputs_all = df_concat_top_5_outputs_all.to_numpy()
np.save(top_5_accuracy_saving_path_npy,np_df_concat_top_5_outputs_all)

# SA dataset and Top-5 accuracy ... mearge them......

df_merged_SA_pred_top_5 = pd.concat([df_merge_SA_pred_logs, df_concat_top_5_outputs_all], axis=1)
#df_merged_SA_pred_top_5.drop(['Class_pred'], axis=1)
top_5_outputs_saving_path_csv = os.path.join(output_path,'merge_SA_prediction_Logs_top_15_accuracy_GLR_1024.csv')
top_5_outputs_saving_path_npy = os.path.join(output_path,'merge_SA_prediction_Logs_top_15_accuracy_GLR_1024.npy')
df_merged_SA_pred_top_5.to_csv(top_5_outputs_saving_path_csv)
np_df_merged_SA_pred_top_5 = df_merged_SA_pred_top_5.to_numpy()
np.save(top_5_outputs_saving_path_npy,np_df_merged_SA_pred_top_5)

# Feature matrix ....
dlfv_rep_all_sp = os.path.join(output_path,'dlfv_all_pred.npy')
np.save(dlfv_rep_all_sp,dlfv_rep_all)



