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
from sklearn.preprocessing import LabelBinarizer

global ip_pos
global set_rows
global test_rows
#
#
# Trains a CNN.
#
# process is not used
#
### Process image
#
#	That's preprocessing not used
#

def process(image_data):
    #clahe = cv2.createCLAHE(2.0, (6, 6))

    for idx, x in enumerate(image_data):

        img = cv2.cvtColor(image_data[idx], cv2.COLOR_BGR2YUV)
        #channels = cv2.split(img)

        #img = cv2.equalizeHist(channels[0])
        #img = clahe.apply(img)
        #channels[0] = img
        #img = cv2.merge(channels)

        #aux = cv2.GaussianBlur(img, (5, 5), 0)
        #img = cv2.addWeighted(img, 1., aux, -0.9, 0)

        #aux = cv2.GaussianBlur(img, (7, 7), 0)
        #img = cv2.addWeighted(img, 1., aux, 0.9, 0)

        img = (img - 128) / 255

        image_data[idx] = img


### Loading Data.
# First loads images
# Corrects from camera
# Computes mirrors

# dirlist
# is a list of directories with data
# builds 1 list of samples [filename, label]
#
#   A on label pot ser car
#
#	It is a recursive search

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

#
#	Labels are set with the vehicles or non-vehicels directories
def buildDirectory(dirlist):

    rows = []
    for dirname in dirlist:
        # Load vehicles
        vehicle_dir = dirname + "/vehicles"
        load_one_directory(vehicle_dir, [1, 0], rows)
        nonvehicle_dir = dirname + "/non-vehicles"
        load_one_directory(nonvehicle_dir, [0, 1], rows)

    return train_test_split(rows, test_size=0.2)

##
#
#	Sample generator

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

training_data = ["./Udacity_Data"]


directory = "./logs/keras_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = directory+"/"+"model.hf5"
if not os.path.exists(directory):
    os.makedirs(directory)
# Create a callback to save best fit

k_callback = ModelCheckpoint(model_filename, monitor='val_acc', verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

# Define some parameters
f_1 = 60
f_2 = 100
f_21 = 250
f_22 = 250
f_3 = 200
f_4 = 50
f_5 = 2
dropout = 0.5
use_valid_data = False
lr = 0.0003
reg = 0.005
epochs = 15

# Build model

model = Sequential()

model.add(Conv2D(f_1, (5, 5), input_shape=(64, 64, 3),
                 activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(f_2, (3, 3),   activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(f_21, (3, 3),   activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))

# Added new layes. Results????
model.add(Conv2D(f_22, (3, 3),   activation='relu', padding='same',
                 kernel_initializer='truncated_normal',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(f_3, kernel_initializer='truncated_normal',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(f_4, kernel_initializer='truncated_normal',
                bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(f_5, kernel_regularizer=regularizers.l2(reg),
                kernel_initializer='truncated_normal',
                bias_initializer='zeros'))
model.add(Activation('softmax'))


# Print some info

print("Dataset", training_data)
print("Saving model data to " + directory)
print("Using Validation Data", use_valid_data)
print("Regularization at Dense(f_5 {:0.2f})".format(reg))
print("Sizes", f_1, f_2, f_21, f_22, f_3, f_4, f_5, "Dropout", dropout, "Regularization", reg, "same")
print("Learning Rate ", lr)

# Train the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Load the data and train the model


batch_size = 256

train_samples, validation_samples = buildDirectory(training_data)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size, 
		    verbose=2, epochs=epochs)

model.save(model_filename)
