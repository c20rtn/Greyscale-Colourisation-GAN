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






# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss,disc_loss


fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

gen_loss_list = []
disc_loss_list = []

for image_batch in dataset:
    t = train_step(image_batch)
    gen_loss_list.append(t[0])
    disc_loss_list.append(t[1])

g_loss = sum(gen_loss_list) / len(gen_loss_list)
d_loss = sum(disc_loss_list) / len(disc_loss_list)

epoch_elapsed = time.time()-epoch_start
print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {hms_string(epoch_elapsed)}')
save_images(epoch,fixed_seed)

elapsed = time.time()-start
print (f'Training time: {hms_string(elapsed)}')





# #Training loop for the GAN
# for epoch in range(EPOCHS):
#     print("EPOCH : ", epoch)
#     start = time.time()
#     #Train the detective
#     for i in range(D_STEPS):
#         #Train on real data batches
#         print(epoch, " : Training on real data")
#         D.fit(X_train_AB[(epoch*D_STEPS)+i],
#                             np.ones(1)) #one label is generated
#         # 
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
