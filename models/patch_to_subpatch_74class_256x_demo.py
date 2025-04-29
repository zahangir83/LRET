#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:37:37 2024

@author: malom
"""
import numpy as np
import pandas as pd
import pdb
from wsi_utils import wsi_utils_functions


patches_source = '/scratch_space/malom/project_74_classes/db_final_20K_final/train/'
patches_saving_dir = '/scratch_space/malom/project_74_classes/db_final_20K_final/train_256x/'
patch_size = (256,256)

pdb.set_trace()

#patches_saving_dir_final =  wsi_utils_functions.patch2subpatches_driver_subdir_KDB_LCDB_testing(patches_source, patches_saving_dir,patch_size)

# Only for the ADIPOSE ...... 
# patches_saving_dir_final =  wsi_utils_functions.patch2subpatches_driver_subdir_KDB_LCDB_testing_ADIPOSE(patches_source, patches_saving_dir,patch_size)

wsi_utils_functions.patch2subpatches_driver_subdir_74class_without_any_condition(patches_source, patch_size, patches_saving_dir)