#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last use: 7 Nov  2018

@author: Rona
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
last use on  nov 18 15:27:46 2017

@author: Rona
"""

import numpy as np
import os
import skimage.io


def convert_arr_to_img():
    
    img_array = np.load("results.npy")
    img_id=np.load("test_id.npy")

    output_dir= "./predicted_masks"
            
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
        
    for i,img_name in zip(range(len(img_array)),img_id):
            mask = img_array[i,:,:,0]
#            .astype('uint64') 
#            mask = img_array[i,:,:,0].astype('uint64')     
            img_file_name = img_name  + '.png'
                
            full_path = os.path.join(output_dir, img_file_name)
            skimage.io.imsave(full_path,mask)
            

    
    
    