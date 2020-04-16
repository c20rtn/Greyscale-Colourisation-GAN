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
import math
from IPython import display
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
EPOCHS = 70
BUFFER_SIZE = 5000
BATCH_SIZE = 16
D_STEPS = 20
G_STEPS = 15

def get_images():
    print("\nGetting Images")
    X = []
    files = glob.glob(DATA_PATH)
    for filename in files:
        X.append(img_to_array(load_img(filename)))
    X = np.array(X, dtype=np.uint8)
    print("\nData", X.shape)
    
    # labels = np.array([os.path.basename(os.path.dirname(image)) for image in files])
    # unique_labels = np.unique(labels).tolist()
    # print(unique_labels)
    # y = labels
    # y = np.array(list(map(lambda x: unique_labels.index(x), labels)))
    # print(np.unique(y))
    
    return X

def dataset2lab(data):
    dataset = data.astype(np.float)
    for i in range(dataset.shape[0]):
        dataset[i] = rgb2lab(dataset[i]/255.0)
    return dataset

def get_test_image():
    testimages = glob.glob(TEST_PATH)
    testimage = img_to_array(load_img(testimages[0]))
    testimage = rgb2lab(1.0/255*testimage)[:,:,0]
    testimage = np.expand_dims(testimage, axis=0)
    testimage = np.expand_dims(testimage, axis=-1)
    # testimage = testimage.reshape(256, 256, 1)
    print("Test image -",testimage.shape)
    return testimage

def make_generator():
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

def make_discriminator():
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

    # cur = np.zeros((256, 256, 3))
    # cur[:,:,0] = X_train_L[0,:,:,0]
    # cur[:,:,1:] = X_train_AB[0,:,:,:]
    # plt.imshow(lab2rgb(cur))
    # plt.show()
    
    print("X_train_L shape : ", X_train_L.shape)
    print("X_test_L shape : ", X_test_L.shape)
    print("X_train_AB shape : ", X_train_AB.shape)
    print("X_test_AB shape : ", X_test_AB.shape)
    return X_train_L, X_test_L, X_train_AB, X_test_AB

def make_image(grayscale, generated):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = generated[0][:,:,0]
    cur[:,:,1:] = grayscale[0]
    return lab2rgb(cur)

X = get_images()
X = dataset2lab(X)
T = get_test_image()
G = make_generator()
D = make_discriminator()

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train_L, X_test_L, X_train_AB, X_test_AB = train_test_LAB_values(X_train, X_test)

del X_train, X_test #Delete unused variables

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=G,
                                discriminator=D)


@tf.function
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = G(T, training=True)

        real_output = D(images, training=True)
        fake_output = D(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, G.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, D.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for i in range(50):
            image_batch = dataset[(i*D_STEPS):(i*D_STEPS)+D_STEPS]
            train_step(image_batch)

        # for image_batch in dataset:
        #     train_step(image_batch)
        

        # Produce images for the GIF as we go
        # display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                         epoch + 1,
        #                         seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('\nTime for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    
    #Output image
    print("\nOutput colorizations")
    
    output = G.predict(T)
    output *= 128
    
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = T[0,:,:,0]
    cur[:,:,1:] = output[0]
    plt.imshow(lab2rgb(cur)) 
    plt.show()

train(X_train_AB, EPOCHS)

#Training loop for the GAN
# for epoch in range(EPOCHS):
#     print("EPOCH : ", epoch)
#     start = time.time()
#     #Train the detective
#     for i in range(D_STEPS):
#         #Train on real data batches
#         print(epoch, " : Training on real data")
#         D.fit(X_train_AB[(epoch*D_STEPS)+i],
#                             np.ones(1)) #one label is generated
        
#         #Train on fake data batches
#         print(epoch, " : Training on generated data")
#         D.fit(G.predict(X_train_L[(epoch*D_STEPS)+i]), 
#                             np.zeros(1)) #zero label is generated

#     #Train the forger
#     for epoch in range(G_STEPS):
#         #Train the detective
#         #Train on real data batches
#         print(epoch, " : Training on real data")
#         D.fit(batch_thingy, train_labels)
    
#     print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))