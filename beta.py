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
EPOCHS = 30
BATCH_SIZE = 16
EPOCH_STEPS = 16

# Get images
testimages = glob.glob(TEST_PATH)
X = []
files = glob.glob(DATA_PATH)
for filename in files:
    X.append(img_to_array(load_img(filename)))
X = np.array(X, dtype=np.uint8)

print("\nData", X.shape)

print("\nSet up train and test data")
# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain
print("\nXtrain", Xtrain.shape)

print("\nCreating Model")
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

print("\nImage transformer")
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

print("\nGenerate training data")
# Generate training data
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

print("\nTrain model")
# Train model      
tensorboard = TensorBoard(log_dir="output\\first_run")

print("\nfit_generator")
model.fit_generator(image_a_b_gen(BATCH_SIZE), callbacks=[tensorboard], epochs=EPOCHS, steps_per_epoch=EPOCH_STEPS)

print("\nTest images")
# Test images
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE))

for img in testimages:
    testimage = img_to_array(load_img(img))
    testimage = rgb2lab(1.0/255*testimage)[:,:,0]

    output = model.predict(testimage)
    output *= 128

    print("\nShow colorizations")

    # L_layer = testimage[:,:,:,0]
    # A_layer = output[:,:,:,0]
    # B_layer = output[:,:,:,1]
    # print("\nL_layer", L_layer.shape)
    # print("\nA_layer", A_layer.shape)
    # print("\nB_layer", B_layer.shape)

    # Output colorizations
    print("\nOutput colorizations")

    timestr = time.strftime("%Y%m%d-%H%M%S")

    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = testimage[0]
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
    ax[1].set_title('Colourised')

    ax[2].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,0)) 
    ax[2].axis('off')
    ax[2].set_title("L: lightness B/W")

    ax[3].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,1)) 
    ax[3].axis('off')
    ax[3].set_title("A: green to red")

    ax[4].imshow(extract_single_dim_from_LAB_convert_to_RGB(cur,2)) 
    ax[4].axis('off')
    ax[4].set_title("B: blue to yellow")

    plt.show()

# imsave("result\\alpha\\"+"img_result.png"+timestr+".png", lab2rgb(cur))
# imsave("result\\alpha\\"+"img_gray_version.png"+timestr+".png", rgb2gray(lab2rgb(cur)))