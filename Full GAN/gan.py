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

DATA_PATH  = '..\\Final Year Project\\Datasets\\cvcl.mit.edu\\coast\\*.jpg'
TEST_PATH  = '.\\Full GAN\\test\\beach.jpg'
EPOCHS = 20
BATCH_SIZE = 20
DISC_SHAPE = (256,256,2)
GEN_SHAPE = (256,256,1)
TRAIN_SET = 400

class GAN():
    def __init__(self):
        ############### Making the Generator ###############
        self.gen = self.make_generator()
        
        ############### Making the Discriminator ###############
        self.disc = self.make_discriminator()
        print("\nDiscriminator Training is OFF")
        self.disc.trainable = False
        
        ############### Making the Arrays for Loss Plot ###############
        self.gen_losses = []
        self.disc_real_losses = []
        self.disc_fake_losses=[] 
        self.disc_acc = []
    
    def get_images(self):
        print("\nGetting Images")
        X = []
        files = glob.glob(DATA_PATH)
        for filename in files:
            X.append(img_to_array(load_img(filename)))
        X = np.array(X, dtype=np.uint8)
        print("\nData", X.shape)
        
        T = img_to_array(load_img(TEST_PATH))
        T = np.array(T, dtype=np.uint8)
        T = rgb2lab(T/255.0)
        ab = T[:,:,1:]
        T = T[:,:,0] / 50 - 1
        T = T.reshape(1, 256, 256, 1)
        
        if len(X) > 4000:
            print("\nTrim dataset")
            X = X[:4000]
        else:
            print("\nDataset not trimmed")
        
        print("X : ", X.shape)
        print("Test : ", T.shape)
        
        return self.dataset2lab(X),T

    def dataset2lab(self, data):
        print("Converting to L*A*B* layers")
        dataset = data.astype(np.float)
        for i in range(dataset.shape[0]):
            if(i % 50 == 0):
                print("Converted file ",i,"/",dataset.shape[0])
            dataset[i] = rgb2lab(dataset[i]/255.0)
        return dataset
    
    def train_test_LAB_values(self):
        images, T = self.get_images()
        
        print("\nSplitting the L* layer")
        X = images[:,:,:,0]
        X = X / 50 - 1
        X = np.expand_dims(X, axis=-1)

        print("Splitting the A*B* layers")
        Y = images[:,:,:,1:] 
        Y = (Y + 128) / 255 * 2 - 1
        
        assert X[0,:,:].shape == (256, 256, 1), "Should be (n, 256, 256, 1)"
        assert Y[0,:,:,:].shape == (256, 256, 2), "Should be (n, 256, 256, 2)"
        
        print("Train test split")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        
        print("X_train_L shape : ", X_train.shape)
        print("X_train_AB shape : ", X_test.shape)
        print("X_test_L shape : ", y_train.shape)
        print("X_test_AB shape : ", y_test.shape)
        
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), T

    def make_generator(self):
        print("\nCreating Generator")
        model = Sequential()
        
        model.add(Conv2D(64,(3,3),padding='same',strides=2, input_shape=GEN_SHAPE))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128, (3,3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(256, (3,3),padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2D(512,(3,3),padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2DTranspose(256,(3,3), strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(128,(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(32,(3,3),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2,(3,3),padding='same'))
        model.add(Activation('tanh'))
        
        gen_input = Input(shape=GEN_SHAPE)
        out = model(gen_input)
        
        print("\nCreated Generator")
        return k.Model(gen_input, out)

    def make_discriminator(self):
        print("\nCreating Discriminator")
        model = Sequential()
        model.add(Conv2D(32,(3,3), padding='same',strides=2,input_shape=DISC_SHAPE))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(64,(3,3),padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(.2))
        model.add(Conv2D(128,(3,3), padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(256,(3,3), padding='same',strides=2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        dis_input = Input(shape=DISC_SHAPE)
        out = model(dis_input)
        
        print("\nCreated Discriminator")
        
        dis = k.Model(dis_input, out)
        dis.compile(loss='binary_crossentropy', 
                        optimizer=Adam(lr=0.00008,beta_1=0.5,beta_2=0.999), 
                        metrics=['accuracy']) 
        
        return dis

    def train_gan(self, X_train, X_test, y_train, y_test, test_image):
        #for file saving
        timestr = time.strftime("%d%m%y-%H%M")
        
        print("\nDefining the combined model")
        gan_input = Input(shape=GEN_SHAPE)
        gen_output = self.gen(gan_input) 
        disc_output = self.disc(gen_output)
        gan_network = k.Model(gan_input, disc_output) 
        
        gan_network.compile(loss='binary_crossentropy', 
                                optimizer=Adam(lr=0.0001,beta_1=0.5,beta_2=0.999))

        y_fake = np.zeros([TRAIN_SET//2,1])
        y_real = np.ones([TRAIN_SET//2,1])
        y_gen = np.ones([TRAIN_SET,1])


        print("\nSTART TRAINING")
        for epoch in range(1,EPOCHS+1):
            start = time.time()
            
            print("\nEpoch - ", epoch, " - SHUFFLE L AND AB CHANNELS")
            np.random.shuffle(X_train)
            l = X_train[:TRAIN_SET]
            np.random.shuffle(y_train)
            ab = y_train[:TRAIN_SET//2]
            
            print("\nEpoch - ", epoch, " - PREDICT FAKE IMAGES")
            fake_images = self.gen.predict(l[:TRAIN_SET//2], verbose=1)
            
            print("\nEpoch - ", epoch, " - Train on Real AB channels")
            d_loss_real = self.disc.fit(x=ab, y= y_real,batch_size=BATCH_SIZE,epochs=1,verbose=1) 
            self.disc_real_losses.append(d_loss_real.history['loss'][-1])
            
            print("\nEpoch - ", epoch, " - Train on fake AB channels")
            d_loss_fake = self.disc.fit(x=fake_images,y=y_fake,batch_size=BATCH_SIZE,epochs=1,verbose=1)
            self.disc_fake_losses.append(d_loss_fake.history['loss'][-1])
            
            print("\nEpoch - ", epoch, " - append the loss and accuracy")
            self.disc_acc.append(d_loss_fake.history['accuracy'][-1])

            print("\nEpoch - ", epoch, " - Train the gan")
            g_loss = gan_network.fit(x=l, y=y_gen,batch_size=BATCH_SIZE,epochs=1,verbose=1)
            
            print("\nEpoch - ", epoch, " - append and print generator loss")
            self.gen_losses.append(g_loss.history['loss'][-1])

            print ('\nTime for epoch {} is {} sec'.format(epoch, time.time()-start))
            
            if epoch % 5 == 0:
                print('Reached epoch:',epoch)
                # predict colourisations
                pred = self.gen.predict(test_image)
                pred = (pred +1) / 2 * 255 - 128
                
                # create image array of 0s
                img = np.zeros((256, 256, 3))
                
                # fill in image array
                img[:,:,0] = (test_image[0][:,:,0] + 1) * 50
                img[:,:,1:] = pred[0]
                
                # save the image
                imsave(".\\Full GAN\\result\\" + timestr + "_epoch_" + str(epoch) + ".png", lab2rgb(img))
                
                if epoch % 10 == 0:
                    self.gen.save('.\\Full GAN\\checkpoints\\generator_' + str(epoch)+ '_' + timestr +'.h5')
        
        #save generator
        self.gen.save('.\\Full GAN\\models\\GAN-' + timestr +'.h5')
        tfjs.converters.save_keras_model(self.gen, '.\\Full GAN\\models\\js')

        #create plots for the losses and accuracy
        plt.plot(self.disc_real_losses, label='Discriminator real')
        plt.plot(self.disc_fake_losses, label='Discriminator fake')
        plt.plot(self.gen_losses, label='Generator')
        plt.savefig(".\\Full GAN\\result\\" + timestr + "-plot.png")


if __name__ == "__main__":
    gan = GAN()
    X_train, X_test, y_train, y_test, T = gan.train_test_LAB_values()
    gan.train_gan(X_train, X_test, y_train, y_test, T)