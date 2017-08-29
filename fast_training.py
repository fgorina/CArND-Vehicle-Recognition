import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
import time
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from utilities import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

### Train and save a svc




## get_filenames buils a list of all vehicle and non_vehicle filenames
#
#   glob is not recursive

def get_filenames(dir):

    filelist = []

    for f in os.listdir(dir):
        if os.path.isdir(dir + "/" + f):
            filelist += get_filenames(dir + "/" + f)
        elif f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"):
            filelist.append(dir + "/" + f)

    return filelist


def get_features(filelist):

    features = []

    for f in filelist:
        features.append(extract_img_features(f))

    return np.array(features)


data_directory = "../Data/"
car_directory = data_directory + "vehicles"
notcar_directory = data_directory + "non-vehicles"
morenotcar_directory = data_directory + "false_positives"
morecar_directory = data_directory + "partial_positives"


C = 2
gamma = 0.00001
kernel = 'linear'
test_percent = 0.2


directory = "./svc_models/svc_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = directory+"_model.pkl"
scaler_filename = directory+"_scaler.pkl"

print("SVC fitting with kernel = {} c = {} and gamma = {}".format(kernel, C, gamma))
print("Data stored in {} files".format(directory))

car_filelist = get_filenames(car_directory)
car_features = get_features(car_filelist)

notcar_filelist = get_filenames(notcar_directory)
notcar_features = get_features(notcar_filelist)

false_filelist = get_filenames(morenotcar_directory)
false_features = get_features(false_filelist)

positives_filelist = get_filenames(morecar_directory)
positives_features = get_features(positives_filelist)

print("Total samples of cars ", len(car_filelist))
print("Total samples of not cars", len(notcar_filelist))
print("Total samples of false positives", len(false_filelist))
print("Total samples of unrecognized positives", len(positives_filelist))
print("Percentage of test samples {:0.2f}".format(test_percent))


#partial_filelist = get_filenames(morecar_directory)
#partial_features = get_features(partial_filelist)

X = np.concatenate((car_features, notcar_features, false_features, positives_features), axis=0).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

with open(scaler_filename, "wb") as f:
    pickle.dump(X_scaler, f)


# Labels Vector
y = np.hstack((np.ones(len(car_features)),
               np.zeros(len(notcar_features)),
               np.zeros(len(false_features)),
               np.ones(len(positives_features))))

# split and shuffle
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=test_percent, random_state=rand_state)

#svc = LinearSVC()

# Train it

svc = SVC(kernel=kernel, C=5, gamma=0.00005, verbose=True)

# Check the training time for the SVC
print("Training...")
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print("Checking...")
accuracy = svc.score(X_test, y_test)
print("Time to train", t2-t, "Accuracy ", round(accuracy, 4))
print(svc.get_params())
with open(model_filename, "wb") as f:
    pickle.dump(svc, f)
