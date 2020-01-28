#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:59:21 2017

@author: Rhona
"""
import os
import numpy as np
import copy

import imageio 
from skimage.draw import set_color
from skimage import color
import skimage.transform 





def visualize_segmentation( ):

    y_pred_path = "./predicted_masks/"
    y_pred_list = os.listdir(y_pred_path)

    img_plus_ypred_WP =  os.path.join(  "./img_plus_ypred/")
    if not os.path.exists(img_plus_ypred_WP):
        os.mkdir(img_plus_ypred_WP)
        
        
    for ypred_name, c in zip(y_pred_list, range(len(y_pred_list))):  # Segmented_Res:
        org_img_name = ypred_name
        
        org_img = imageio.imread( "./Data/test_image/" + org_img_name)
        y_pred = imageio.imread(os.path.join(y_pred_path, ypred_name))
    
        y_pred= skimage.transform.resize(y_pred, (org_img.shape[0], org_img.shape[1]))
        
        org_img = color.gray2rgb(org_img)
    
        
        nonzero_y_pred = np.nonzero(y_pred)
        

        img_plus_y_pred = copy.deepcopy(org_img)
        set_color(img_plus_y_pred, nonzero_y_pred, [255, 0, 0])
    
        
        imageio.imwrite(img_plus_ypred_WP + ypred_name , img_plus_y_pred)



