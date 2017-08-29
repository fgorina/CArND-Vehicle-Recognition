import numpy as np
import random
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import os
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

global ip_pos
global set_rows
global test_rows

### Same as learn_cars but instead of defining the model uses an aready trained one

## Preprocessing data - Not Used
#
def process(image_data):
    clahe = cv2.createCLAHE(2.0, (6, 6))

    for idx, x in enumerate(image_data):

        img = cv2.cvtColor(image_data[idx], cv2.COLOR_BGR2YUV)
#        channels = cv2.split(img)

#        img = cv2.equalizeHist(channels[0])
#        img = clahe.apply(img)
#        channels[0] = img
#        img = cv2.merge(channels)

#        aux = cv2.GaussianBlur(img, (5, 5), 0)
#        img = cv2.addWeighted(img, 1., aux, -0.9, 0)

#        aux = cv2.GaussianBlur(img, (7, 7), 0)
#        img = cv2.addWeighted(img, 1., aux, 0.9, 0)

        img = (img - 128) / 255

        image_data[idx] = img


### Loading Data.
# First loads images
# Corrects from camera
# Computes mirrors

# dirlist is a list of directories with data
# builds 1 list of itels [filename, label]
#
#   A on label pot ser car

def get_filenames(dir, label):

    filelist = []

    for f in os.listdir(dir):
        if os.path.isdir(dir + "/" + f):
            filelist += get_filenames(dir + "/" + f, label)
        elif f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"):
            filelist.append([dir + "/" + f, label])

    return filelist


def load_one_directory(dirname, label, rows):
    names = get_filenames(dirname, label)
    rows.extend(names)


def buildDirectory(dirlist):

    rows = []
    for dirname in dirlist:
        # Load vehicles
        vehicle_dir = dirname + "/vehicles"
        load_one_directory(vehicle_dir, [1, 0], rows)
        nonvehicle_dir = dirname + "/non-vehicles"
        load_one_directory(nonvehicle_dir, [0, 1], rows)

    return train_test_split(rows, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            labels = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                if image.shape[0] != 64 or image.shape[1] != 64:
                    image = cv2.resize(image, (64, 64))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image = (image - 128) / 255
                label = batch_sample[1]
                images.append(image)
                labels.append(label)

            # trim image to only see section with road
            X_train = np.array(images)
            # Normalize between -1 and 1


            y_train = np.array(labels)
            yield (X_train, y_train)

### training directory

training_data = ["./Data"]


directory = "./logs/keras_2017-08-28_10-59-48"
model_filename = directory+"/"+"model.hf5"
new_model_filename = directory+"/updated_model.hf5"



# Print some info

print("Dataset", training_data)

# Train the model

model = load_model(model_filename)
# Load the data and train the model


batch_size = 256
epochs=15

train_samples, validation_samples = buildDirectory(training_data)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size, epochs=epochs)

model.save(new_model_filename)
