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

DATA_PATH  = '.\\cvcl.mit.edu\\coast\\n203015.jpg'
TEST_PATH  = '.\\Alpha\\test\\*.jpg'
EPOCHS = 100

# Get images
def get_images():
    image = img_to_array(load_img(DATA_PATH))
    image = np.array(image, dtype=np.uint8)
    T = glob.glob(TEST_PATH)

    X = rgb2lab(image/255.0)[:,:,0]
    Y = rgb2lab(image/255.0)[:,:,1:]
    Y /= 128
    X = X.reshape(1, 256, 256, 1)
    Y = Y.reshape(1, 256, 256, 2)
    
    assert X.shape == (1, 256, 256, 1), "Should be (1, 256, 256, 1)"
    assert Y.shape == (1, 256, 256, 2), "Should be (1, 256, 256, 2)"
    
    print("X size - ", X.shape)
    print("Y size - ", Y.shape)
    print("test images size - ", len(T), " - ", T[0])
    return X, Y, T

def create_generator():
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
    model.compile(optimizer='rmsprop',loss='mse')
    return model

def train_gen(model, X, Y):
    
    assert X.shape == (1, 256, 256, 1), "Should be (1, 256, 256, 1)"
    assert Y.shape == (1, 256, 256, 2), "Should be (1, 256, 256, 2)"
    
    model.fit(x=X, 
        y=Y,
        batch_size=1,
        epochs=EPOCHS)
    print(model.evaluate(X, Y, batch_size=1))
    
    return model

def output_colourisations(model, test):
    count = 0
    for img in test:
        print(img)
        count = count + 1
        testimage = img_to_array(load_img(img))
        testimage = rgb2lab(testimage/255.0)
        testimage = testimage[:,:,0]
        testimage = testimage.reshape(1, 256, 256, 1)

        assert testimage.shape == (1, 256, 256, 1), "Should be (1, 256, 256, 1)"
        
        output = model.predict(testimage)
        output *= 128
        
        assert output.shape == (1, 256, 256, 2), "Should be (1, 256, 256, 2)"

        print("\nShow colorizations")

        L_layer = testimage[:,:,:,0]
        A_layer = output[:,:,:,0]
        B_layer = output[:,:,:,1]
        print("\nL_layer", L_layer.shape)
        print("\nA_layer", A_layer.shape)
        print("\nB_layer", B_layer.shape)

        # Output colorizations
        print("\nOutput colorizations")

        timestr = time.strftime("%Y%m%d-%H%M%S")

        cur = np.zeros((256, 256, 3))
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

        plt.show()
        # fig.tight_layout()
        # plt.savefig('first-result-'+str(count)+'.png', bbox_inches='tight')

print(tf.__version__,"\n\n\n")
print(tfjs.__version__,)
X, Y, testimages = get_images()
gen = create_generator()
gen = train_gen(gen, X, Y)

# gen = tf.keras.models.load_model('.\\Models\\Alpha\\alpha.h5')

# Check its architecture
gen.summary()

output_colourisations(gen, testimages)

# gen.save('.\\Alpha\\Models\\model.h5')
# tfjs.converters.save_keras_model(gen, '.\\Models\\Alpha\\js')