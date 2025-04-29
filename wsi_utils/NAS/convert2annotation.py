"""
Convert histopathology mask image to SVG annotation

Usage:
    convert2annotation [steatosis|inflammation|fibrosis|ballooning] [--color <color>] [--width <width>] [--fill] [--closed] <filepath>
    convert2annotation (-h | --help)
    convert2annotation --version

Options:
    -h --help                                      Show help.
    --version                                      Show version.
    --color <color>                                Override a color of segmentation.
    --width <width>                                Stroke width [default: 3].
    --fill                                         Fill segmentation objects [default: False].
    --closed                                       Close segmentations curves [default: False].
"""

import cv2
import math
import numpy as np
import os
import pyvips

MIN_ADJ = 0.000001
MAX_ADJ = 0.1

if __name__ == '__main__':
    from docopt import docopt
    arguments = docopt(__doc__, version='{}'.format(1.0))

    if arguments['--color']:
        color=arguments['<color>']
    else:
        if arguments['steatosis']:
            color='red'
        elif arguments['inflammation']:
            color='blue'
        elif arguments['ballooning']:
            color='green'
        elif arguments['fibrosis']:
            color='yellow'
        else:
            color='black'

    stroke_width = arguments['--width']
    fill = arguments['--fill']
    closed = arguments['--closed']

    if arguments['steatosis']:
        tissue_class = 'steatosis'
    elif arguments['inflammation']:
        tissue_class = 'inflammation'
    elif arguments['ballooning']:
        tissue_class = 'ballooning'
    elif arguments['fibrosis']:
        tissue_class = 'fibrosis'
    else:
        print('ERROR! A valid tissue is required!')
        exit(1)

    if arguments['<filepath>']:
        input_mask_file_path = arguments['<filepath>']
    print(input_mask_file_path)
    input_mask_file_path_tokens = input_mask_file_path.split('/')
    if (len(input_mask_file_path_tokens) == 0):
        input_mask_file_path_tokens = input_mask_file_path.split('\\')

    output_mask_file_path = input_mask_file_path_tokens[0]
    for token in range(1, len(input_mask_file_path_tokens) - 1):
        output_mask_file_path = output_mask_file_path + '\\' + input_mask_file_path_tokens[token]

    output_mask_filename = input_mask_file_path_tokens[-1]
    output_mask_filename = output_mask_filename.split('.')[0]
    output_mask_filename = output_mask_filename + '.svg'
    output_mask_file_path = output_mask_file_path + '\\' + output_mask_filename

    # Read the image
    image = pyvips.Image.new_from_file(input_mask_file_path, access='sequential')
    imgray = np.ndarray(buffer=image.write_to_memory(),
                         dtype=np.uint8,
                         shape=[image.height, image.width, image.bands])

    # Find image size
    print(imgray.shape)
    height = imgray.shape[0]
    width = imgray.shape[1]

    # Find number of decimals saved in SVG file
    decimals_height = len(str(height)) - 2
    decimals_width = len(str(width)) - 2

    # Find number of patches (Divide the image into patches to avoid memory overflow)
    image_block_size = 64000
    patch_width = 1
    patch_height = 1
    if height * width > 2 * 1024**3:
        patch_height = math.ceil(height / image_block_size)
        patch_width = math.ceil(width / image_block_size)

    height_chunck = math.floor(height / patch_height)
    width_chunck = math.floor(width / patch_width)
    contours_list = []
    origin_pos = np.zeros((1,1,2))
    for i in range (patch_height):
        for j in range (patch_width):
            f_i = i*height_chunck 
            f_j = j*width_chunck

            if i == patch_height:
                l_i = height - f_i
            else:
                l_i = (i+1)*height_chunck

            if j == patch_width:
                l_j = width - f_j
            else:
                l_j = (j+1)*width_chunck

            img_patch = imgray[ f_i: l_i, f_j:l_j, :]
            greyscale_img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(greyscale_img_patch,127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            new_contours = []
            origin_pos[0, 0, 0] = f_j
            origin_pos[0, 0, 1] = f_i
            for k in contours:
                k[:, :, :] = k[:, :, :] + origin_pos
                new_contours.append(k)

            contours_list.extend(new_contours)

    f = open(output_mask_file_path, 'w+')
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" preserveAspectRatio="none" viewBox="0 0 100 100" width="100%" height="100%">')
    

    for elem in contours_list:
        if elem.shape[0] > 2: # Skip shapes with less than 3 points
            f.write(f'<path d="') # begin of a path
            for p in range(elem.shape[0]):
                x = round( 100 * elem[p, 0, 0] / width, decimals_height)
                y = round( 100 * elem[p, 0, 1] / height, decimals_width)
                if p == 0: # first node in a pathmust start with M
                    f.write(f'M{x} {y}')
                else: # the rest of points
                    # x_1 = round( 100 * elem[p - 1, 0, 0] / width, decimals_height)
                    # y_1 = round( 100 * elem[p - 1, 0, 1] / height, decimals_width)
                    # if (abs(x - x_1) > MIN_ADJ) and (abs(y - y_1) > MIN_ADJ) and ((abs(x - x_1) < MAX_ADJ) or (abs(y - y_1) < MAX_ADJ)):
                    f.write(f'L{x} {y}')
                    if  p == elem.shape[0] - 1:
                        if closed: # verify whether it is the last point in a path
                            f.write(' z')
            f.write('"')
            f.write(f' class="{tissue_class}"') # end of points
            # if fill:
            #     f.write(f' fill="{color}"')
            # else:
            #     f.write(f' fill="none"')
            f.write('></path>') # end of a path
    f.write('</svg>')
    f.close()
