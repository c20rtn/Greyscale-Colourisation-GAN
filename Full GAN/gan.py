from __future__ import absolute_import, division, print_function, unicode_literals
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import glob
import numpy as np
import os
import random
import time
import cv2
import tqdm
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

DATA_PATH  = '..\\Final Year Project\\Datasets\\cvcl.mit.edu\\**\\*.jpg'
TEST_PATH  = '.\\Full GAN\\test\\beach.jpg'
EPOCHS = 150
BUFFER_SIZE = 5000
BATCH_SIZE = 20
DISC_SHAPE = (256,256,2)
GEN_SHAPE = (256,256,1)
TRAIN_SET = 320

# class GAN():
#     def __init__(self):
#         self.discriminator = self.make_discriminator()
#         self.generator = self.make_generator()
        

def get_images():
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
    T = T[:,:,0]
    T = T.reshape(256, 256, 1)
    # labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
    # unique_labels = np.unique(labels).tolist()
    # print(unique_labels)
    # y = labels
    # y = np.array(list(map(lambda x: unique_labels.index(x), labels)))
    # print(np.unique(y))
    
    print("X : ", X.shape)
    print("Test : ", T.shape)
    return dataset2lab(X),T

def train_test_LAB_values(input):
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

def dataset2lab(data):
    print("Converting to L*A*B* layers")
    dataset = data.astype(np.float)
    for i in range(dataset.shape[0]):
        if(i % 50 == 0):
            print("Converted file ",i,"/",dataset.shape[0])
        dataset[i] = rgb2lab(dataset[i]/255.0)
    return dataset

def make_generator():
    print("\nCreating Generator")
    model = Sequential()
    
    model.add(Conv2D(64,(3,3),padding='same',strides=2, input_shape=GEN_SHAPE))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #128 x 128
    
    model.add(Conv2D(128, (3,3), padding='same',strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #64 x 64
    
    model.add(Conv2D(256, (3,3),padding='same',strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #32 x 32 
    
    model.add(Conv2D(512,(3,3),padding='same',strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #16 x 16
    
    
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
    
    l_channel = Input(shape=GEN_SHAPE)
    image = model(l_channel)
    print("\nCreated Generator")
    return k.Model(l_channel,image)

def make_discriminator():
    print("\nCreating Discriminator")
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same',strides=2,input_shape=DISC_SHAPE))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,(3,3),padding='same',strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(.2))
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(128,(3,3), padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(256,(3,3), padding='same',strides=2))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    image = Input(shape=DISC_SHAPE)
    validity = model(image)
    print("\nCreated Discriminator")
    return k.Model(image,validity)

#for file saving
timestr = time.strftime("%d%m%y-%H%M")

X, T = get_images()
X_train, X_test, y_train, y_test = train_test_LAB_values(X)

del X

############### Making the Discriminator ###############
dis = make_discriminator()
dis.compile(loss='binary_crossentropy', 
                    optimizer=Adam(lr=0.00008,beta_1=0.5,beta_2=0.999), 
                    metrics=['accuracy']) 

#Making the Discriminator untrainable so that the generator can learn from fixed gradient 
print("\nDiscriminator Training is OFF")
dis.trainable = False

############### Making the Generator ###############
gen = make_generator()

#Defining the combined model of the Generator and the Discriminator 
print("\nDefining the combined model")
l_channel = Input(shape=GEN_SHAPE)
image = gen(l_channel) 
valid = dis(image)

combined_network = k.Model(l_channel, valid) 
combined_network.compile(loss='binary_crossentropy', 
                        optimizer=Adam(lr=0.0001,beta_1=0.5,beta_2=0.999))

#creates lists to log the losses and accuracy
print("\nlosses and accuracy")
gen_losses = []
disc_real_losses = []
disc_fake_losses=[] 
disc_acc = []

#train the generator on a full set of 320 and the discriminator on a half set of 160 for each epoch
#discriminator is given real and fake y's while generator is always given real y's

y_train_fake = np.zeros([TRAIN_SET/2,1])
y_train_real = np.ones([TRAIN_SET/2,1])
y_gen = np.ones([TRAIN_SET,1])

#run and train until photos meet expectations (stop & restart model with tweaks if loss goes to 0 in discriminator)
print("\nSTART TRAINING")
for epoch in range(1,EPOCHS+1):
    #shuffle L and AB channels then take a subset corresponding to each networks training size
    print("\nEpoch - ", epoch, " - SHUFFLE L AND AB CHANNELS")
    np.random.shuffle(X_train)
    l = X_train[:TRAIN_SET]
    np.random.shuffle(y_train)
    ab = y_train[:160]
    
    print("\nEpoch - ", epoch, " - PREDICT FAKE IMAGES")
    fake_images = gen.predict(l[:160], verbose=1)
    
    #Train on Real AB channels
    print("\nEpoch - ", epoch, " - Train on Real AB channels")
    d_loss_real = dis.fit(x=ab, y= y_train_real,batch_size=BATCH_SIZE,epochs=1,verbose=1) 
    disc_real_losses.append(d_loss_real.history['loss'][-1])
    
    #Train on fake AB channels
    print("\nEpoch - ", epoch, " - Train on fake AB channels")
    d_loss_fake = dis.fit(x=fake_images,y=y_train_fake,batch_size=BATCH_SIZE,epochs=1,verbose=1)
    disc_fake_losses.append(d_loss_fake.history['loss'][-1])
    
    #append the loss and accuracy and print loss
    print("\nEpoch - ", epoch, " - append the loss and accuracy")
    disc_acc.append(d_loss_fake.history['accuracy'][-1])
    

    #Train the gan by producing AB channels from L
    print("\nEpoch - ", epoch, " - Train the gan")
    g_loss = combined_network.fit(x=l, y=y_gen,batch_size=BATCH_SIZE,epochs=1,verbose=1)
    #append and print generator loss
    print("\nEpoch - ", epoch, " - append and print generator loss")
    gen_losses.append(g_loss.history['loss'][-1])

    #every 50 epochs it prints a generated photo and every 100 it saves the model under that epoch
    if epoch % 25 == 0:
        print('Reached epoch:',epoch)
        pred = gen.predict(T.reshape(1,256,256,1))
        img = lab2rgb(np.dstack((T,pred.reshape(256,256,2))))
        img = np.array(img)
        imsave(".\\Full GAN\\result\\" + timestr + "_epoch_" + str(epoch) + ".png", img)
        
        if epoch % 50 == 0:
            gen.save('.\\Full GAN\\checkpoints\\generator_' + str(epoch)+ '_' + timestr +'.h5')

# gen.save('.\\Full GAN\\models\\GAN.h5')
# tfjs.converters.save_keras_model(gen, '.\\Full GAN\\models\\js')

plt.plot(disc_real_losses)
plt.plot(disc_fake_losses)
plt.plot(gen_losses)
plt.savefig(".\\Full GAN\\result\\" + timestr + "-plot.png")

#print the generated image
pred = gen.predict(T.reshape(1,256,256,1))
final = lab2rgb(np.dstack((T,pred.reshape(256,256,2))))
plt.imshow(final)