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
from sklearn.model_selection import train_test_split
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

# DATA_PATH  = '..\\Final Year Project\\Datasets\\cvcl.mit.edu\\*\\*.jpg'
DATA_PATH  = '..\\Final Year Project\\Datasets\\cvcl.mit.edu\\coast\\*.jpg'
TEST_PATH  = '.\\Beta\\test\\*.jpg'
EPOCHS = 100
BATCH_SIZE = 8

def get_images():
    print("\nGetting Images")
    X = []
    files = glob.glob(DATA_PATH)
    for filename in files:
        X.append(img_to_array(load_img(filename)))
    X = np.array(X, dtype=np.uint8)
    print("\nData", X.shape)
    
    T = glob.glob(TEST_PATH)
    
    return dataset2lab(X),T

def train_test_LAB_values(input):
    print("\nSplitting the L* layer")
    X = input[:,:,:,0]
    X = np.expand_dims(X, axis=-1)

    print("Splitting the A*B* layers")
    Y = input[:,:,:,1:] 
    Y /= 128
    
    assert X[0,:,:].shape == (256, 256, 1), "Should be (n, 256, 256, 1)"
    assert Y[0,:,:,:].shape == (256, 256, 2), "Should be (n, 256, 256, 2)"
    
    print("Train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    print("X_train_L shape : ", X_train.shape)
    print("X_train_AB shape : ", X_test.shape)
    print("X_test_L shape : ", y_train.shape)
    print("X_test_AB shape : ", y_test.shape)
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def dataset2lab(data):
    print("Converting to L*A*B* layers")
    dataset = data.astype(np.float)
    for i in range(dataset.shape[0]):
        if(i % 50 == 0):
            print("Converted file ",i,"/",dataset.shape[0])
        dataset[i] = rgb2lab(dataset[i]/255.0)
    
    assert len(dataset) == len(data), "Should be equal"
    
    return dataset

def create_gen():
    print("\nCreating Model")
    # Building the neural network
    model = k.Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))
    
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    
    # Finish model
    model.compile(optimizer='rmsprop', loss='mse')
    
    return model

def fit_gen(model, Xtrain, Xtest, Ytrain, Ytest):

    print("\nTrain model")
    model.fit(  x=Xtrain, 
                y=Ytrain,
                batch_size=BATCH_SIZE,
                validation_data = (Xtest, Ytest),
                steps_per_epoch = len(Xtrain)/BATCH_SIZE,
                epochs = EPOCHS)        
    
    return model

def output_colourisations(model, testimages):
    count = 0
    for img in testimages:
        count = count + 1
        testimage = img_to_array(load_img(img))
        testimage = rgb2lab(testimage/255.0)[:,:,0]
        testimage = testimage.reshape(1, 256, 256, 1)

        output = model.predict(testimage)
        output *= 128

        print("\nShow colorizations")

        L_layer = testimage[:,:,:,0]
        A_layer = output[:,:,:,0]
        B_layer = output[:,:,:,1]
        print("\nL_layer", L_layer.shape)
        print("\nA_layer", A_layer.shape)
        print("\nB_layer", B_layer.shape)

        # Output colorizations
        timestr = time.strftime("%Y%m%d-%H%M%S")

        cur = np.zeros((256, 256, 3), dtype=np.float)
        cur[:,:,0] = testimage[0][:,:,0]
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

        ax[0].imshow(mpimg.imread(img)) 
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
        plt.savefig('beta-result-'+str(count)+'.png', bbox_inches='tight')

X, testimages = get_images()
X_train, X_test, y_train, y_test = train_test_LAB_values(X)

del X

gen = create_gen()
gen = fit_gen(gen, X_train, X_test, y_train, y_test)

output_colourisations(gen, testimages)

gen.save('.\\Beta\\Models\\beta.h5')
tfjs.converters.save_keras_model(gen, '.\\Beta\\Models\\JS')