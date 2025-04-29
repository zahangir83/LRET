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


# Path for 415 cases .......

SA_dataset_path = ''

###### >>>>>>>>>>>>>>>>LOAD the Final Survival and Labels datasets... <<<<<<<<<<<<<<<<<<<<<<
survival_logs_with_labels415 = np.load(join_path(SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.npy'),allow_pickle=True)
pd_survival_logs_with_labels415 = pd.read_csv(join_path(SA_dataset_path,'Final_415case_IDS_with_labels_for_BRAIN.csv'))

pdb.set_trace()

directory_case_ids = pd_survival_logs_with_labels415['patient_ID']
