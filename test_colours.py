from __future__ import absolute_import, division, print_function, unicode_literals
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import glob
import numpy as np
import os
import random
import time
import cv2
import sys
import matplotlib as mat
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflowjs as tfjs
import tensorflow.keras as k
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

TEST_PATH  = '.\\Full GAN\\test\\*'
EPOCHS = 500

def output_colourisations(model, test):
    count = 0
    for img in test:
        print(img)
        count = count + 1
        testimage = img_to_array(load_img(img))
        testimage = rgb2lab(1.0/255*testimage)
        testimage = testimage[:,:,0]
        testimage = testimage / 50 - 1
        testimage = testimage.reshape(1, 256, 256, 1)

        output = model.predict(testimage)
        output = (output +1) / 2 * 255 - 128

        print("\nShow colorizations")

        timestr = time.strftime("%Y%m%d-%H%M%S")

        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = (testimage[0][:,:,0] + 1) * 50
        cur[:,:,1:] = output[0]

        def extract_single_dim_from_LAB_convert_to_RGB(image,idim):
            '''
            image is a single lab image of shape (None,None,3)
            '''
            z = np.zeros(image.shape)
            if idim != 0 :
                z[:,:,0]=80 ## I need brightness to plot the image along 1st or 2nd axis
            z[:,:,idim] = image[:,:,idim]
            z = lab2rgb(z)
            return(z)

        fig, ax = plt.subplots(1, 5, figsize = (16, 6))
        
        if img != ".\Full GAN\\test\sample1.jpg":
            ax[0].imshow(mpimg.imread(img)) 
            ax[0].axis('off')
            ax[0].set_title('Original')
        else:
            ax[0].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,0)) 
            ax[0].axis('off')
            ax[0].set_title('Original')

        ax[1].imshow(lab2rgb(cur)) 
        ax[1].axis('off')
        ax[1].set_title('Lab scaled')

        ax[2].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,0)) 
        ax[2].axis('off')
        ax[2].set_title("L: lightness")

        ax[3].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,1)) 
        ax[3].axis('off')
        ax[3].set_title("A: green to red")

        ax[4].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,2)) 
        ax[4].axis('off')
        ax[4].set_title("B: blue to yellow")

        #plt.show()
        fig.tight_layout()
        plt.savefig('.\\Full GAN\\temp\\gan-result-'+str(count)+'.png', bbox_inches='tight')

T = glob.glob(TEST_PATH)

gen = tf.keras.models.load_model('.\\Models\\GAN\\gan.h5')
# Check its architecture
gen.summary()

output_colourisations(gen, T)