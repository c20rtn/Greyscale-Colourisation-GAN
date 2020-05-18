# Set Up for Colourisation

Change the `TEST` and `DATA` paths in the code to where the 256*256 data resides.

Change the `EPOCHS` based on how many iterations of training are need.

Change `BATCH` size based on RAM size (I did 20 batch size on 12gb of ram).

Required Libraries:

- Tensorflow
- tensorflowjs
- skimage
- sklearn
- matplotlib
- cv2
- numpy


# Alpha Generation

The first iteration of the generator model revolves around the experimentation of the simple use of a Keras model and how that could be used togenerate colours for any input with the generator model trained on one image.  
This is to simply attempt to create a working generator model, see how certain parameters make the results differ and create baseline functions for the full GAN.

---

# Beta Generation

The second implementation of the generator revolves around the generators fitting of multiple images, in this case the expanse of a whole data-set. This will let the convolutional generator model learn multiple image scenarios (such as forests, cities and beaches).

---

# GAN Generation

Opposing layout of the first and second iterations of generators, the GAN Python code will be structured as a class that can be called, initialised and class methods can be called from it.
This allows the full network to be wrapped, possibly imported into another Python file and more easily tested due to separation of functions for easier debugging.

---

# Website

[Colourisation Website](https://c20rtn.github.io/)