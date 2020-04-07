from __future__ import absolute_import, division, print_function, unicode_literals
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import glob
import numpy as np
import os
import random
import time
import cv2
import matplotlib as mat
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
import tensorflow as tf
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

DATA_PATH  = '..\\Final Year Project\\Datasets\\cvcl.mit.edu\\coast\\*.jpg'
TEST_PATH  = 'test\\*.jpg'
EPOCHS = 100
BATCH_SIZE = 16
EPOCH_STEPS = 16

def generator():
    print("Creating Generator")
    
def discriminator():
    print("Creating Generator")