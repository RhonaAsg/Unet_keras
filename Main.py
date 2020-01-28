#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update on Sat Apr 13 16:31:10 2019

@author: Rhona

training with 2 classes (box, background)


The parameters you need to change if you using the code for another dataset:
image_rows
image_rows 
num_epoch
batch_size
data_path

"""
from Util.my_Unet import Unet_model
from Util.prepare_data import prepare_data
from Util.convert_arr_to_img import convert_arr_to_img
from Util.postprocessing import visualize_segmentation
# =============================================================================
# Variables
image_rows = 512
image_cols = 1024
num_epoch = 70
batch_size= 14
data_path= "./Data"
# =============================================================================
#%%  1) CONVERT IMAGES TO NUMPY ARRAY
prepare_data (image_rows, image_cols, data_path, "train")
prepare_data (image_rows, image_cols, data_path, "test" )

# =============================================================================
#%% 2) TRAIN AND TEST THE NETWORK
Unet_model(batch_size, num_epoch,image_rows,image_cols)

# =============================================================================
#%%
convert_arr_to_img()
#%%
visualize_segmentation()