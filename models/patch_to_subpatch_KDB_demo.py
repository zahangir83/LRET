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


patches_source = '/scratch_space/malom/project_74_classes/LC_KDB_testing_from_74CLSDB/database_1024/adipose_1024/'
patches_saving_dir = '/home/malom/stjude_projects/computational_pathology/Large_scale_histopathology_analysis/database/LC_KDB_testing_from_74CLSDB/adipose_512x_patches/adipose/'
patch_size = (512,512)

pdb.set_trace()

#patches_saving_dir_final =  wsi_utils_functions.patch2subpatches_driver_subdir_KDB_LCDB_testing(patches_source, patches_saving_dir,patch_size)

# Only for the ADIPOSE ...... 
patches_saving_dir_final =  wsi_utils_functions.patch2subpatches_driver_subdir_KDB_LCDB_testing_ADIPOSE(patches_source, patches_saving_dir,patch_size)

