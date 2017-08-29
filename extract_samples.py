import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
import cv2
import os
from utilities import *
from glob import glob

### This program processes all images in a directory with a sliding window algorithm
#
#	and writes the 64x64 images to a directory

def output_windows(file, output_dir, window_size=64, scale=1):

    img = cv2.imread(file)
    name = file.split("/")[-1]
    if not name.endswith("jpg") and not name.endswith(".jpg") and not name.endswith(".png"):
        return

    just_name = name.split(".")[0]

    ystart = 0
    ystop = 0

    if scale != 1:
        imshape = img.shape
        ctrans_tosearch = cv2.resize(img, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
    else:
        ctrans_tosearch = np.copy(img)

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    pixels_per_step = 16

    nxsteps = (ctrans_tosearch.shape[1] - window + pixels_per_step) // pixels_per_step
    nysteps = (ctrans_tosearch.shape[0] - window + pixels_per_step) // pixels_per_step

    box_list = []
    all_box_list = []
    image_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * pixels_per_step
            xpos = xb * pixels_per_step

            xbox_left = np.int(xpos * scale)
            ytop_draw = np.int(ypos * scale)
            win_draw = np.int(window * scale)

            subimg = ctrans_tosearch[ypos:ypos + window, xpos:xpos + window]

            window_name = output_dir + "/" + just_name + str(yb) + "_" + str(xb)+ ".jpg"

            cv2.imwrite(window_name, subimg)

    return

# Load an image
# scan at different scales
# store every 64x64 patch in a directory

directory = "../Paco/non_images"
output_dir = "../Paco/reconegut/non-vehicles"

for file in os.listdir(directory):
    filename = directory + "/" + file
    output_windows(filename, output_dir, window_size=64, scale=1)
    output_windows(filename, output_dir, window_size=64, scale=1.5)
