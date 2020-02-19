from __future__ import absolute_import, division, print_function, unicode_literals

import os
from os import listdir
import glob
import random
import matplotlib as mat
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model

from PIL import Image
from pathlib import Path
import pprint
pp = pprint.PrettyPrinter(indent=4)

#CONSTANTS
EPOCHS = 25
DATA_PATH  = '../Final Year Project\Datasets\cvcl.mit.edu\**\*.jpg'
BUFFER_SIZE = 60000

#get dataset
files = glob.glob(DATA_PATH)

#pre-process data
images = np.array([np.array(Image.open(image)) for image in files])
labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
unique_labels = np.unique(labels).tolist()
#put data into one structure

print(unique_labels)
print("Training Images shape : ", images.shape)

X = images
#y = labels
y = np.array(list(map(lambda x: unique_labels.index(x), labels)))
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X shape : ", X.shape)
print("y shape : ", y.shape)
print("Split X shape : ", X_train.shape, X_test.shape)
print("Split y shape : ", y_train.shape, y_test.shape)

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(X_train[i])
#     plt.xlabel(y_train[i])
#     plt.ylabel(unique_labels[y_train[i]])
# plt.show()

def make_discriminator():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(256, 256, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=EPOCHS)

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    return model 

discriminator  = make_discriminator()

def make_generator_model():
    """
    Returns generator as Keras model.
    """
    input = Input(shape=(256, 256, 1))
    
    conv = Conv2D(64, (3,3), activation="relu")(input)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128, (3,3), activation="relu")(input)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(64, (3,3), activation="relu")(input)
    conv = BatchNormalization()(conv)

    model = Model(inputs=input,outputs=conv)
    return model


generator = make_generator_model()
img = mpimg.imread('test.png')
print(img)
plt.imshow(img)