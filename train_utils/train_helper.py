#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:32:28 2023

@author: malom
"""

import os
import glob
#from sklearn.utils import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pdb

def file_counter(dir, output=False):
    files_per_folder = {}

    if not os.path.exists(dir):
        raise Exception("{} does not exist".format(dir))

    folders = next(os.walk(dir))[1]
    for f in folders:
        files = glob.glob(os.path.join(dir, f, '*.jpg'))
        files_per_folder[f] = len(files)
        if output:
            print("{} : {}".format(f, len(files)))

    return files_per_folder

def file_counter_png(dir, output=False):
    files_per_folder = {}

    if not os.path.exists(dir):
        raise Exception("{} does not exist".format(dir))

    folders = next(os.walk(dir))[1]
    for f in folders:
        files = glob.glob(os.path.join(dir, f, '*.png'))
        files_per_folder[f] = len(files)
        if output:
            print("{} : {}".format(f, len(files)))

    return files_per_folder

def get_class_weights(folder):
    '''
    Given a folder (containing class name folders), returns a dict containing how much emphasis should be
    placed on each class (less files = higher weight)

    :param folder:
    :return:
    '''
    file_count = file_counter_png(folder)
    folders = next(os.walk(folder))[1]

    # list containing each folder name occuring the same number of times as its file count
    temp = []
    for f in folders:
        temp += [f] * file_count[f]
    
    #pdb.set_trace()
    
    #res = compute_class_weight('balanced', np.array(folders), temp)
    train_labels = folders
    res = compute_class_weight(class_weight = "balanced", classes= np.unique(train_labels), y= temp)

    res = dict(enumerate(res))

    return res
