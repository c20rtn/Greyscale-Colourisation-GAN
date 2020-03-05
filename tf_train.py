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
from skimage import io, color

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

from PIL import Image
from pathlib import Path
import pprint
from keras.layers.pooling import AveragePooling2D
pp = pprint.PrettyPrinter(indent=4)

#CONSTANTS
EPOCHS = 20
DATA_PATH  = '../Final Year Project\Datasets\cvcl.mit.edu\**\*.jpg'
BUFFER_SIZE = 60000

class GAN():
    def __init__(self):
        """
        Initialize the GAN. Includes compiling the generator and the discriminator separately and then together as the GAN.
        """
        self.generator_input = (256,256,1) #used for the L layer of L*A*B* (grayscale image)
        self.discriminator_input = (256,256,2) #used for the A and B layers of L*A*B*
        
        #Create the generator 
        self.generator = self.make_generator()
        g_opt = Adam(lr=.001)
        self.generator.compile(loss='binary_crossentropy', optimizer=g_opt)
        print('Generator Summary...')
        print(self.generator.summary())
        
        #Create the classifier/discrimiator 
        self.discriminator = self.make_discriminator()
        d_opt = Adam(lr=.0001)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])
        print('Discriminator Summary...')
        print(self.discriminator.summary())
        
        
        gan_input = Input(shape=self.g_input_shape) #Give an input shape to the GAN
        img_color = self.generator(gan_input) #generator of input shape
        
        #Need to make this false as the discrimiator will automatically scale itself and make itself a fairer adversary thus ruining the GAN
        self.discriminator.trainable = False 
        
        real_or_fake = self.discriminator(img_color) #
        self.gan = Model(gan_input,real_or_fake) #
        opt = Adam(lr=.001)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt) #Compiles the gan 
        print('\n')
        print('GAN summary...')
        print(self.gan.summary())

    def make_discriminator(self):
        # model = keras.Sequential([
        #     keras.layers.Flatten(input_shape=self.discriminator_input),
        #     keras.layers.Dense(128, activation='relu'),
        #     keras.layers.Dense(len(unique_labels), activation='softmax')
        # ])

        # model.compile(optimizer='adam',
        #             loss='sparse_categorical_crossentropy',
        #             metrics=['accuracy'])

        # model.fit(X_train, y_train, epochs=EPOCHS)

        # test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
        # print('\nTest accuracy:', test_acc)
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.d_input_shape, strides=2))
        model.add(LeakyReLU(.2))
        # model.add(Dropout(.25))

        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same',strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding='same',strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.25))

        # model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Dropout(.5))

        # model.add(Conv2D(512, (3, 3), padding='same',strides=2))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(.2))
        # model.add(Dropout(.25))

        model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Dropout(.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def make_generator(self):
        
        g_input = Input(shape=self.generator_input)
        
        conv = Conv2D(64, (3,3), activation="relu")(input)
        conv = BatchNormalization()(conv)
        
        conv = Conv2D(128, (3,3), activation="relu")(conv)
        conv = BatchNormalization()(conv)
        
        conv = Conv2D(64, (3,3), activation="relu")(conv)
        conv = BatchNormalization()(conv)

        model = Model(inputs=g_input,outputs=conv)
        return model

    def train_discriminator(self, X_train_L, X_train_AB, X_test_L, X_test_AB):
        """
        Function to train the discriminator. Called when discriminator accuracy falls below and a specified threshold.
        """
        generated_images = self.generator.predict(X_train_L)
        X_train = np.concatenate((X_train_AB, generated_images))
        n = len(X_train_L)
        y_train = np.array([[1]] * n + [[0]] * n)
        rand_arr = np.arange(len(X_train))
        np.random.shuffle(rand_arr)
        X_train = X_train[rand_arr]
        y_train = y_train[rand_arr]

        test_generated_images = self.generator.predict(X_test_L)
        X_test = np.concatenate((X_test_AB, test_generated_images))
        n = len(X_test_L)
        y_test = np.array([[1]] * n + [[0]] * n)
        rand_arr = np.arange(len(X_test))
        np.random.shuffle(rand_arr)
        X_test = X_test[rand_arr]
        y_test = y_test[rand_arr]

        self.discriminator.fit(x=X_train, y=y_train, epochs=1)
        metrics = self.discriminator.evaluate(x=X_test, y=y_test)
        print('\n accuracy:',metrics[1])
        if metrics[1] < .90:
            self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
    
    def train(self, X_train_L, X_train_AB, X_test_L, X_test_AB, epochs):
        """
        Training loop for GAN. First the discriminator is fit with real and fake images. Next the Generator is fit. This is possible because the weights in the Discriminator are fixed and not affected by back propagation.
        Inputs: X_train L channel, X_train AB channels, X_test L channel, X_test AB channels, number of epochs.
        Outputs: Models are saved and loss/acc plots saved.
        """

        # self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
        g_losses = []
        d_losses = []
        d_acc = []
        X_train = X_train_L
        n = len(X_train)
        y_train_fake = np.zeros([n,1])
        y_train_real = np.ones([n,1])
        for e in range(epochs):
            #generate images
            np.random.shuffle(X_train)
            generated_images = self.generator.predict(X_train, verbose=1)
            np.random.shuffle(X_train_AB)

            #Train Discriminator
            d_loss  = self.discriminator.fit(x=X_train_AB, y=y_train_real,  batch_size=16, epochs=1)
            if e % 3 == 2:
                noise = np.random.rand(n,256,256,2) * 2 -1
                d_loss = self.discriminator.fit(x=noise, y=y_train_fake, batch_size=16, epochs=1)
            d_loss = self.discriminator.fit(x=generated_images, y=y_train_fake, batch_size=16, epochs=1)
            d_losses.append(d_loss.history['loss'][-1])
            d_acc.append(d_loss.history['acc'][-1])
            print('d_loss:', d_loss.history['loss'][-1])
            # print("Discriminator Accuracy: ", disc_acc)

            #train GAN on grayscaled images , set output class to colorized
            g_loss = self.gan.fit(x=X_train, y=y_train_real, batch_size=16, epochs=1)

            #Record Losses/Acc
            g_losses.append(g_loss.history['loss'][-1])
            print('Generator Loss: ', g_loss.history['loss'][-1])
            disc_acc = d_loss.history['acc'][-1]

            # Retrain Discriminator if accuracy drops below .8
            if disc_acc < .8 and e < (epochs / 2):
                self.train_discriminator(X_train_L, X_train_AB, X_test_L, X_test_AB)
            if e % 5 == 4:
                print(e + 1,"batches done")

        self.plot_losses(g_losses,'Generative Loss', epochs)
        self.plot_losses(d_acc, 'Discriminative Accuracy',epochs)
        self.generator.save('../models/gen_model_full_batch_' + str(epochs)+'.h5')
        self.discriminator.save('../models/disc_model_full_batch_' + str(epochs)+'.h5')

    def plot_losses(self, metric, label, epochs):
        """
        Plot the loss/acc of the generator/discriminator.
        Inputs: metric, label of graph, number of epochs (for file name)
        """
        plt.plot(metric, label=label)
        plt.title('GAN Accuracy and Loss Over ' + str(epochs) + ' Epochs')
        plt.savefig('../plots/plot_' + str(epochs) + '_epochs.png')
        # plt.close()

if __name__ == '__main__':
    
    #get dataset
    files = glob.glob(DATA_PATH)

    #pre-process data
    images = np.array([np.array(Image.open(image).convert('RGB')) for image in files])
    labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
    unique_labels = np.unique(labels).tolist()
    #put data into one structure

    print(unique_labels)
    print("Training Images shape : ", images.shape)

    X = images
    #y = labels
    # y = np.array(list(map(lambda x: unique_labels.index(x), labels)))
    # print(np.unique(y))
    #print("SAMPLE IMAGE ARRAY", X[1])
    # print("Converting from RGB to L*A*B*")
    # for img in X: 
    #     img = color.rgb2lab(img)
    
    print("Splitting test and train")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("X shape : ", X.shape)
    # print("y shape : ", y.shape)
    print("Split X shape : ", X_train.shape, X_test.shape)
    # print("Split y shape : ", y_train.shape, y_test.shape)

    print("Splitting the L* layer")
    X_train_L = np.array([i[:, :, 0] for i in X_train])
    X_test_L = np.array([i[:, :, 0] for i in X_test])
    print("X_train_L shape : ", X_train_L.shape)
    print("X_test_L shape : ", X_test_L.shape)
    
    print("Splitting the A*B* layers")
    X_train_AB = np.zeros((X_train.shape[0],256,256,2), 'uint8')
    X_test_AB = np.zeros((X_test.shape[0],256,256,2), 'uint8')
    X_train_AB[..., 0] = [i[:, :, 1] for i in X_train]
    X_train_AB[..., 1] = [i[:, :, 2] for i in X_train]
    X_test_AB[..., 0] = [i[:, :, 1] for i in X_test]
    X_test_AB[..., 1] = [i[:, :, 2] for i in X_test]
    
    # X_train_AB = np.dstack((np.array([i[:, :, 1] for i in X_train]),np.array()))
    # X_test_AB = np.array([[i[:, :, 1],i[:, :, 2]] for i in X_test])
    print("X_train_AB shape : ", X_train_AB.shape)
    print("X_test_AB shape : ", X_test_AB.shape)
    
    # gan = GAN()
    # gan.train(X_train_L, X_train_AB, X_test_L, X_test_AB, EPOCHS)
    
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
    