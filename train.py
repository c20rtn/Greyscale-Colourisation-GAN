import os
from os import listdir
import glob
import random
import matplotlib as mat
from matplotlib.image import imread
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

#CONSTANTS
epochs = 50
dataset_path = 'Datasets/cvcl.mit.edu/coast/*.jpg'

#get dataset
images = []
files = glob.glob(dataset_path)

#pre-process data
train_images = np.array([np.array(Image.open(image)) for image in files])
print("Training Images shape : ", train_images.shape)

#train the model
def discriminator_model():
    model = tf.keras.Sequential()