# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:08:43 2019
@author: deeplens
"""
import numpy as np
import os
import locate_tissue 
import pdb


abspath = os.path.dirname(os.path.abspath(__file__))

slide_path = '/home/mza/Desktop/MedPace_projects/steatosis_detection_project/steatosis_seg_project/database/wsis/test/122R0VUJ.tif'


slide_path_final = os.path.join(abspath,slide_path)  
# locate tissue contours with default parameters


""" 
    Locate tissue contours of whole slide image
    Parameters
    ----------
    slide_path : valid slide path
        The slide to locate the tissue.
    max_img_size: int
        Max height and width for the size of slide with selected level.
    smooth_sigma: int
        Gaussian smoothing sigma.
    thresh_val: float
        Thresholding value.
    min_tissue_size: int
        Minimum tissue area.
    
    
    Returns
    -------
    cnts: list
        List of all contours coordinates of tissues.
    d_factor: int
        Downsampling factor of selected level compared to level 0
    
"""

#pdb.set_trace()

slide_image, binary_image, cnts, d_factor = locate_tissue.locate_tissue_cnts(slide_path_final, max_img_size=2048, smooth_sigma=13, 
                                       thresh_val=0.80,min_tissue_size=10000)
                                       
print('Downsampling factor is: '+ str(d_factor))

# Draw contour on the images...


