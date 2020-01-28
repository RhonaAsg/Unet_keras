#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:06:13 2017

@author: Rhona
"""
from __future__ import print_function

import os
import numpy as np
from skimage.transform import resize
import imageio


def prepare_data(image_rows, image_cols, data_path, split_name):
    print('-------------converting tif images to numpy array---------')
        
    # GET THE INPUT IMAGES AND MASK PATH
    image_path = os.path.join(data_path,split_name + "_image")
    
    # GET THE MASK DATA PATH IF IT IS TRAINING MODE
    if split_name== "train":
      mask_path  = os.path.join(data_path,split_name+  "_mask")
    
    
    imgs_id   = []
    images    = os.listdir(image_path)
    total     = len(images) 
    indx_cntr = 0

#%% ITERATE ON NUMBER OF THE IMAGES
    imgs = np.ndarray((total, image_rows, image_cols),dtype=np.uint8)
    if split_name== "train":
      imgs_mask = np.ndarray((total , image_rows, image_cols ), dtype=np.uint8)
    
    for image_name in images:
        image_name_split=image_name.split('.')[0]
        
        # READ THE IMAGE 
        img = imageio.imread(os.path.join(image_path, image_name))
        img = resize(img, (image_rows, image_cols), preserve_range=True)
        
        # READ THE MASK IF IT IS TRAIN MODE
        if split_name== "train":
          img_mask = imageio.imread(os.path.join(mask_path, image_name))
          img_mask=resize(img_mask,(image_rows, image_cols), preserve_range=True)
        
        img_id = str(image_name_split)
        imgs_id.append( img_id) 
        
        img = np.array([img])
        imgs[indx_cntr] = img
        if split_name== "train":
          imgs_mask[indx_cntr] = img_mask
    

        if indx_cntr % 20 == 0:
            print('Done: {0}/{1} images'.format(indx_cntr, total))
        indx_cntr += 1
        
#%% SAVING THE NUMPY ARRAYS
    np.save(split_name + '_image.npy', imgs)
    np.save(split_name + '_id.npy', imgs_id)
    if split_name== "train":
      np.save('train_mask.npy', imgs_mask)
    print('-------------Saving as .npy files done-------------') 
    
    
    
def load_train_data():
    # LOADING TRAIN IMAGES AND MASK
    imgs_train = np.load("train_image.npy")
    imgs_mask_train = np.load("train_mask.npy")
    return imgs_train, imgs_mask_train

def load_test_data():
    # LOADING TEST IMAGES
    imgs_test = np.load("test_image.npy")
    return imgs_test