from __future__ import absolute_import, division, print_function, unicode_literals
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import os
import random
import time
import cv2
import math
from IPython import display
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

DATA_PATH  = '..\\Final Year Project\\Datasets\\cvcl.mit.edu\\**\\*.jpg'
TEST_PATH  = 'test\\beach.jpg'
EPOCHS = 70
BUFFER_SIZE = 5000
BATCH_SIZE = 16
D_STEPS = 20
G_STEPS = 15

class GAN():
    def __init__(self):
        self.discriminator = self.make_discriminator()
        self.generator = self.make_generator()
        
        
        
        
        
    
    def get_images(self):
        print("\nGetting Images")
        X = []
        files = glob.glob(DATA_PATH)
        for filename in files:
            X.append(img_to_array(load_img(filename)))
        X = np.array(X, dtype=np.uint8)
        print("\nData", X.shape)
        
        T = glob.glob(TEST_PATH)
        # labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
        # unique_labels = np.unique(labels).tolist()
        # print(unique_labels)
        # y = labels
        # y = np.array(list(map(lambda x: unique_labels.index(x), labels)))
        # print(np.unique(y))
        
        return dataset2lab(X),T

    def train_test_LAB_values(self, input):
        print("\nSplitting the L* layer")
        X = input[:,:,:,0]
        X = np.expand_dims(X, axis=-1)

        print("Splitting the A*B* layers")
        Y = input[:,:,:,1:] 
        
        print("Train test split")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        
        print("X_train_L shape : ", X_train.shape)
        print("X_train_AB shape : ", X_test.shape)
        print("X_test_L shape : ", y_train.shape)
        print("X_test_AB shape : ", y_test.shape)
        print("X_train_L type : ", type(X_train))
        print("X_train_AB type : ", type(X_test))
        print("X_test_L type : ", type(y_train))
        print("X_test_AB type : ", type(y_test))
        
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def dataset2lab(self, data):
        print("Converting to L*A*B* layers")
        dataset = data.astype(np.float)
        for i in range(dataset.shape[0]):
            if(i % 250 == 0):
                print("Converted file ",i,"/",dataset.shape[0])
            dataset[i] = rgb2lab(dataset[i]/255.0)
        return dataset

    def make_generator(self):
        print("\nCreating Generator")
        model = k.Sequential()
        model.add(InputLayer(input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.compile(optimizer='rmsprop', loss='mse')
        
        return model

    def make_discriminator(self):
        print("\nCreating Discriminator")
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256,256,2), strides=2))
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        model.add(Conv2D(64, (3, 3), padding='same',strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        model.add(Conv2D(128, (3, 3), padding='same',strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        model.add(Conv2D(256, (3, 3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.5))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        return model
    
    def train(self):
        

def make_image(grayscale, generated):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = generated[0][:,:,0]
    cur[:,:,1:] = grayscale[0]
    return lab2rgb(cur)

if __name__ == '__main__':
    gan = GAN()