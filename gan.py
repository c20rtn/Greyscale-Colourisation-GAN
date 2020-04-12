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
EPOCHS = 100
BATCH_SIZE = 16
EPOCH_STEPS = 16

def get_images():
    print("\nGetting Images")
    X = []
    files = glob.glob(DATA_PATH)
    for filename in files:
        X.append(img_to_array(load_img(filename)))
    X = np.array(X, dtype=np.uint8)
    print("\nData", X.shape)
    
    labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
    unique_labels = np.unique(labels).tolist()
    print(unique_labels)
    y = labels
    y = np.array(list(map(lambda x: unique_labels.index(x), labels)))
    print(np.unique(y))
    
    return X, y

def dataset2lab(data):
    dataset = data.astype(np.float)
    for i in range(dataset.shape[0]):
        dataset[i] = rgb2lab(dataset[i]/255.0)
    return dataset

def get_test_image():
    testimages = glob.glob(TEST_PATH)
    testimage = img_to_array(load_img(testimages[0]))
    testimage = rgb2lab(1.0/255*testimage)[:,:,0]
    testimage = testimage.reshape(1, 256, 256, 1)

def generator():
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

def discriminator():
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

def train_test_LAB_values(X_train, X_test):
    print("\nSplitting the L* layer")
    X_train_L = X_train[:,:,:,0]
    X_test_L = X_test[:,:,:,0]
    X_train_L = np.expand_dims(X_train_L, axis=-1)
    X_test_L = np.expand_dims(X_test_L, axis=-1)

    print("Splitting the A*B* layers\n")
    X_train_AB = X_train[:,:,:,1:] 
    X_test_AB = X_test[:,:,:,1:] 

    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = X_train_L[0,:,:,0]
    cur[:,:,1:] = X_train_AB[0,:,:,:]
    plt.imshow(lab2rgb(cur))
    plt.show()
    
    print("X_train_L shape : ", X_train_L.shape)
    print("X_test_L shape : ", X_test_L.shape)
    print("X_train_AB shape : ", X_train_AB.shape)
    print("X_test_AB shape : ", X_test_AB.shape)
    return X_train_L, X_test_L, X_train_AB, X_test_AB

if __name__ == '__main__':
    
    print("\nGAN STARTED")
    
    X, y = get_images()
    # T = get_test_image()
    # G = generator()
    D = discriminator()
    
    # X_train, X_test = train_test_split(dataset2lab(X), test_size=0.2, random_state=42)
    print("\nSet up train and test data")
    # Set up train and test data
    split = int(0.8*len(X))
    Xtrain = X[:split]
    Xtrain = 1.0/255*Xtrain
    print("\nXtrain", Xtrain.shape)
    X_train_L, X_test_L, X_train_AB, X_test_AB = train_test_LAB_values(X_train, X_test)
    
    print("\nCOMPILE STARTED")
    D.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    print("\nTRAINING STARTED")
    D.fit(X_train_L, X_train_AB, epochs=10)
    test_loss, test_acc = D.evaluate(X_test_L, X_test_AB, verbose=2)

    print('\nTest accuracy:', test_acc)