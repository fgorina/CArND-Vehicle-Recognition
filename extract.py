import pickle
import random
import cv2
import numpy as np
import csv
from random import shuffle

### To process and extract Udacity Set data
#
#	It is used to generate the CNN training dataset


source_dir = "../object-detection-crowdai/"
source_labels = source_dir + "labels.csv"

output_dir = "./Udacity_Data/"
vehicle_dir = output_dir + "vehicles/"
notvehicle_dir = output_dir + "non-vehicles/"

def intersect(box1, box2):
    if box1[0][0] > box2[1][0] or box1[1][0] < box2[0][0] or box1[0][1] > box2[1][1] or box1[1][1] < box2[0][1] :
        return False
    else:
        return True

def intersect_list(box1, boxlist):

    for box2 in boxlist:
        if intersect(box1, box2):
            return True
    return False

def find_empty_boxes(bigbox, usedboxlist, n):

    emptyboxlist = []

    xmin = bigbox[0][0]
    xmax = bigbox[1][0] - 64

    ymin = bigbox[0][1]
    ymax = bigbox[1][1] - 84  

    x = xmin
    y = ymax

    while y > ymin:
        x = xmin
        found = False
        while x < xmax:
            box = ((x, y), (x+64, y+64))

            if not intersect_list(box, usedboxlist):
                emptyboxlist.append(box)
                x += 64
                found = True
            else:
                x += 1
        if found:
            y -= 64
        else:
            y -= 1

    shuffle(emptyboxlist)
    return emptyboxlist[0:min(n, len(emptyboxlist))]

def write_empty_images(dir, name, img, usedboxlist, n):


    bigbox = ((0, img.shape[0]//2), (img.shape[1], img.shape[0]))

    boxes = find_empty_boxes(bigbox, usedboxlist, n)

    i = 0
    for box in boxes:
        subimg = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        i += 1
        new_filename = dir + str(i) + "_" + name
        cv2.imwrite(new_filename, subimg)

    #print(len(usedboxlist), len(boxes), n)




# Load the data from labels file
old_filename = ""
img = None
boxlist=[]

nimages = 0
maximages = 2000
firstline = True

ncars_image = 0
with open(source_labels) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if firstline:
            firstline = False
            continue

        if nimages > maximages:
            if img is not None:
                write_empty_images(notvehicle_dir, old_filename, img, boxlist, ncars_image)
            break

        if not row[4].endswith(".jpg"):
            continue

        if row[4] != old_filename or img is None:

            if img is not None:
                write_empty_images(notvehicle_dir, old_filename, img, boxlist, ncars_image)

            # Read new image
            boxlist = []
            old_filename = row[4]
            filename = source_dir+row[4]
            img = cv2.imread(filename)
            ncars_image = 0


        label = row[5]

        if label == 'Car' or label == 'Truck':
            xmin = int(row[0])
            ymin = int(row[1])
            xmax = int(row[2])
            ymax = int(row[3])


            subimg = img[ymin:ymax, xmin:xmax]
            dx = xmax-xmin
            dy = ymax-ymin

            if dx < 12 or dy < 12:
                continue

            box = ((xmin, ymin), (xmax, ymax))
            boxlist.append(box)


            if dx > dy:
                new_dx = 64
                new_dy = int(new_dx/dx*dy)
                top_border = int((new_dx - new_dy)/2)
                bottom_border = new_dx - new_dy - top_border
                left_border = 0
                right_border = 0

            else:
                new_dy = 64
                new_dx = int(new_dy/dy*dx)
                top_border = 0
                bottom_border = 0
                left_border = int((new_dy - new_dx)/2)
                right_border = new_dy - new_dx - left_border

            resized = cv2.resize(subimg, (new_dx, new_dy))

            padded = cv2.copyMakeBorder(resized, top_border, bottom_border,
                left_border, right_border, borderType=cv2.BORDER_REPLICATE )

            nimages += 1
            ncars_image += 1

            new_filename = vehicle_dir + str(nimages) + "_" + old_filename

            cv2.imwrite(new_filename, padded)
