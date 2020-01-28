#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last update on Sat Mar 13 22:01:10 2017

@author: Rhona

training a Unet with 2 classes (target and background)
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import CSVLogger
from tensorflow.keras            import backend as K
from Util.prepare_data           import load_train_data, load_test_data
from tensorflow.keras.layers     import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

#MAKE SURE THE CHANNEL IS THE LAST AXIS
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


#%% Unet LOSS FUNCTION
smooth = 1

def dice_coef(y_true, y_pred,eps=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    temp = (K.sum(y_true_f) + K.sum(y_pred_f) + eps)
    temp1= (2 * intersection + eps)
    dice = temp1 / temp
    return dice



def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#%%
def unet_arch(img_rows,img_cols, num_classes=1):
    
    # CONV BLOCK 1
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # CONV BLOCK 2
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # CONV BLOCK 3
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # CONV BLOCK 4
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # CONV BLOCK 5
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    # CONV BLOCK 6
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    # CONV BLOCK 7
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    # CONV BLOCK 8
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    # CONV BLOCK 9
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    # CONV BLOCK 10
    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=1e-4),loss=dice_coef_loss, metrics=[dice_coef])

    return model

#%%
def preprocess(imgs):
    imgs_p = imgs
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

#%%
def train(num_epoch, batch_size, img_rows,img_cols):
    print("-------------Training the Unet-------------")
    # random seed for reproducibility
    np.random.seed(202)
    
    # LOAD TRAIN IMAGES AND MASKS
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = preprocess(imgs_train)   
    
    
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    imgs_mask_train = K.cast(imgs_mask_train, dtype='float32')

    model = unet_arch(img_rows,img_cols)

    logfile_name=os.path.join( 'log.csv')
    csv_logger = CSVLogger(logfile_name, append=False, separator=';')
    history = model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=num_epoch, verbose=1, shuffle=True,
#              validation_split=0.0,validation_data=None,
              validation_split=0.2,
              callbacks=[csv_logger])
    
    model.save_weights('weights.h5',save_format='h5')

#%% uncomment this section in order to plot dice losss plot
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title("model dice_coef_loss")
    plt.ylabel("dice_coef_loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('dice_coef_loss_performance.png')
    plt.clf()
    plt.plot(history.history['dice_coef'], label='train')
    plt.plot(history.history['val_dice_coef'], label='valid')
    plt.title("model dice_coef")
    plt.ylabel("dice_coef")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('dice_coef_performance.png')
    
#%% 
def predict(batch_size,img_rows,img_cols):
    print("-------------- Predicting on test data-------------")
    # LOAD TEST IMAGES 
    imgs_test = load_test_data()
    imgs_test = preprocess(imgs_test)
#    imgs_test = imgs_test.astype(np.float32)
#    imgs_test /= 255
    
    imgs_test = K.cast(imgs_test, dtype='float32')
    model = unet_arch(img_rows,img_cols)
    # LOADING THE TRAINED WEIGHTS
    model.load_weights('weights.h5')
    imgs_mask_test = model.predict(imgs_test,batch_size=batch_size ,verbose=1)
    # SAVING THE PREDICTIONSs
    np.save('results.npy', imgs_mask_test)    
    
def Unet_model(batch_size, num_epoch,img_rows,img_cols):

    train(num_epoch, batch_size,img_rows,img_cols)
    predict(batch_size,img_rows,img_cols)
    