#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:47:17 2020

@author: malom
"""


import numpy as np
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn import preprocessing
from os.path import join as join_path
import pdb

data = np.load('beta_samples_91_classes.npy')
data = np.transpose(data) 
chip_idloc_all = np.load('chip_id_loc.npy')
labels = np.load('actual_labels.npy')    
#idat_idloc = labels[:,0]

print(data.shape)
print(chip_idloc_all.shape)
print(labels.shape)

data = np.array(data)
chip_idloc_all = np.array(chip_idloc_all)
labels = np.array(labels)

selected_beta = []
selected_chip_idloc = []
selected_ac_labels = []
    
k = 0
kk = 0
for i, idv_ac_label in enumerate(labels):
    if  'CONTR-CEBM' in idv_ac_label:
        #selected_beta.append(data[i,:])
        #print('Processing for row number :', str(i))
        selected_beta.append(data[i,:])
        selected_chip_idloc.append(chip_idloc_all[i])
        selected_ac_labels.append(labels[i])
        k = k+1

#pdb.set_trace()
selected_chip_idloc = np.array(selected_chip_idloc)
selected_beta = np.array(selected_beta)
selected_ac_labels = np.array(selected_ac_labels)
print('Number of CONTR',selected_beta.shape[0])

dataset_saving_path = 'CONTR-CEBM/'
fn_selected_beta_path = join_path(dataset_saving_path,'CONTR-CEBM_beta.npy')
fn_selected_idat_path = join_path(dataset_saving_path,'CONTR-CEBM_idat.npy')
fn_selected_ac_labels_path = join_path(dataset_saving_path,'CONTR-CEBM_ac_labels.npy')
       
np.save(fn_selected_beta_path,selected_beta)
np.save(fn_selected_idat_path,selected_chip_idloc)
np.save(fn_selected_ac_labels_path,selected_ac_labels)