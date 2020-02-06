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

from PIL import Image
from pathlib import Path
import pprint
pp = pprint.PrettyPrinter(indent=4)

#CONSTANTS
epochs = 50
dataset_path = '../Final Year Project\Datasets\Places2\data_256/a/auto_showroom/*.jpg'

#get dataset
files = glob.glob(dataset_path)

#pre-process data
images = np.array([np.array(Image.open(image)) for image in files])
labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
unique_labels = np.unique(labels)
#put data into one structure

print(unique_labels)
print("Training Images shape : ", images.shape)

X = images
y = labels

# view an image (e.g. 25) and print its corresponding label
# img_index = 782
# plt.imshow(X[img_index,:,:,:])
# plt.show()
# print(y[img_index])

X = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]*X.shape[3]))
y = y.reshape(y.shape[0],)

print("X shape : ", X.shape)
print("y shape : ", y.shape)

X, y = shuffle(X, y, random_state=42)

clf = RandomForestClassifier()
print(clf)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,preds))