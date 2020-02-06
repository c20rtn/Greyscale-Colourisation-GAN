import os
from os import listdir
import glob
import random
import matplotlib as mat
import matplotlib.pyplot as plt
from matplotlib.image import imread
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

from PIL import Image
from pathlib import Path
import pprint
pp = pprint.PrettyPrinter(indent=4)

#CONSTANTS
EPOCHS = 50
dataset_path = '../Final Year Project\Datasets\cvcl.mit.edu\**\*.jpg'

#get dataset
files = glob.glob(dataset_path)

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

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(y_train[i])
    plt.ylabel(unique_labels[y_train[i]])
plt.show()

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