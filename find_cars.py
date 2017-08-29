import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utilities import *
from glob import glob
# Initial Setup for Keras

from keras.models import load_model

# Used for computing lateral distance

hor_x = 640 # 620
hor_y = 360 # 420


## Class Car represent a possible car

class Car:

    def __init__(self, position, size, color, image):

        # name is given to recognise it
        self.name = None

        # color to draw
        self.drawing_color = (255, 0, 0)

        # position is a tuple (x, y)
        self.position = position

        # size is a tuple (dx, dy)
        self.size = size

        # speed is a tuple in pixels/frame (vx, vy)
        self.speed = None

        # sigma2 position is a tuple (sigma2x, sigma2y) with squared position errors
        self.sigma2 = None

        # color is a BGR color. Averaged over the bounding box
        self.color = color

        # image is last image
        self.image = image

        # frames_seen is how many frames it has been see continuously
        self.frames_seen = 0

        # frames_unseen is how many frames it has been unseen continuouslu
        self.frames_unseen = 0

        # last position is a list with last position.
        # last one is at top
        self.last_positions = []
        self.last_sizes = []
        self.last_speeds = []
        self.last_colors = []


	# Returns distance from a position
    def distance(self, pos):
        return distance(self.position, pos)

	# Returns a new position updated with the speed 
    def updated_position(self):
        if self.speed is not None:
            return (self.position[0] + self.speed[0], self.position[1] + self.speed[1])
        else:
            return self.position

    # Update the car data with that from a posible car,
    #
    #	This one is like a Kalman Filter and has been tried but is not used in the videos
    
    def update_data_kalman(self, car):

        self.last_positions.insert(0, car.position)
        self.last_sizes.insert(0, car.size)

        insta_speed_x = car.position[0]-self.position[0]
        insta_speed_y = car.position[1]-self.position[1]

        # Compute new position

        x = (self.position[0] * car.size[0] + car.position[0]*self.sigma2[0])/(car.size[0] + self.sigma2[0])
        y = (self.position[1] * car.size[1] + car.position[1]*self.sigma2[1])/(car.size[1] + self.sigma2[1])

        sigma2x = 1 / (1/self.sigma2[0] + 1/car.size[0])
        sigma2y = 1 / (1 / self.sigma2[1] + 1 / car.size[1])


        self.speed = (insta_speed_x, insta_speed_y)
        self.position = (x, y)
        self.sigma2 = (sigma2x, sigma2y)

        # car size may be anying mostly at the beginning
        #   Perjaps just made average
        self.size = np.average(self.last_sizes[0:3], axis=0)
        self.frames_seen += 1
        self.frames_unseen = 0  # If we have seen it now it is not here

        d_color = color_difference(self.color, car.color)
        print("Updated Car {} color difference {} Speed {}".format(self.name, d_color, self.speed))

	# Updates car data with new observation
	#
	#	Just uses averages
	
    def update_data(self, car):

        self.last_positions.insert(0, car.position)
        self.last_sizes.insert(0, car.size)
        self.last_colors.insert(0, car.color)

        #self.color = np.average(self.last_colors[0:10], axis=0)


        # Compute new position

        self.position = np.average(self.last_positions[0:3], axis=0)  
        self.size = np.average(self.last_sizes[0:7], axis=0) 

        # Compute new speed
        insta_speed_x = car.position[0]-self.position[0]
        insta_speed_y = car.position[1]-self.position[1]
        self.last_speeds.insert(0, (insta_speed_x, insta_speed_y))

        recent_positions = self.last_positions[0:5]

        if len(recent_positions) >= 2:
            self.speed = ((recent_positions[0][0]-recent_positions[-1][0])/len(recent_positions),
                    (recent_positions[0][1]-recent_positions[-1][1])/len(recent_positions))

        self.frames_seen += min(self.frames_seen+1, 20)
        self.frames_unseen = 0  # If we have seen it now it is not here

        d_color = color_difference(self.color, car.color)
        print("Updated Car {} color difference {} Speed {}".format(self.name, d_color, self.speed))

	# Computes the area of a box
	
    def area(self):
        return self.size[0] * self.size[1]
        
    # Returns a bbox from a car
    
    def car_box(self):
        return bbox_new(self.position, self.size)



    # box_draw tries to compute and draw a "cube" over the car. Not satisfied as some computations
    #	are not correct when near the sides of the image

    def box_draw(self, image, font):

        car_prop = 2  # car length / car width

        # First we compute sinus and cosinus

        x = self.position[0] - hor_x + self.size[0]/2
        y = self.position[1] - hor_y + self.size[1]/2  # compute it at the bottom
        r = np.sqrt(x*x+y*y)
        alfa = x/r
        beta = y/r

        w = self.size[0]
        h = self.size[1]


        car_width = w / (1+car_prop*alfa*beta)
        car_height = h - beta * car_prop * car_width * beta
        car_length = car_prop * car_width


        dx = self.size[0] - car_width
        dy = self.size[1] - car_height
        # Back Box

        bottom = self.position[1] + self.size[1]/2
        left = self.position[0]+(w/2 - car_width)
        top = bottom - car_height
        right = left + car_width


        back_box = ((int(left), int(top) ), (int(right), int(bottom)))

        # Front Box
        front_box = ((int(left-dx), int(top-dy)), (int(right-dx), int(bottom-dy)))

        color = self.drawing_color
        thick = 2
        cv2.rectangle(image, front_box[0], front_box[1], color, thickness=thick)
        cv2.rectangle(image, back_box[0], back_box[1], color, thickness=thick)

        cv2.line(image, front_box[0], back_box[0], color, thickness=thick)
        cv2.line(image, (front_box[1][0], front_box[0][1]), (back_box[1][0], back_box[0][1]), color, thickness=thick)
        cv2.line(image, (front_box[0][0], front_box[1][1]), (back_box[0][0], back_box[1][1]), color, thickness=thick)
        cv2.line(image, (front_box[1][0], front_box[1][1]), (back_box[1][0], back_box[1][1]), color, thickness=thick)



        b = self.car_box()

        #cv2.rectangle(image, b[0], b[1], (255,0,0), thickness=1)


    # box_draw_p is like box_draw but tries a perspective effect

    def box_draw_p(self, image, font):


        car_prop = 2  # car length / car width

        # First we compute sinus and cosinus at midle bottom point (aprox)

        x = self.position[0] - hor_x + self.size[0] / 2
        y = self.position[1] - hor_y + self.size[1] / 2  # compute it at the bottom
        r = np.sqrt(x * x + y * y)
        alfa = x / r
        beta = y / r

        top = int(self.position[1]-self.size[1]/2)
        left = int(self.position[0]-self.size[0]/2)
        bottom = int(self.position[1]+self.size[1]/2)
        right = int(self.position[0]+self.size[0]/2)

        w = self.size[0]
        h = self.size[1]

        car_width = w / (1 + car_prop * alfa * beta)
        dx = w - car_width

        top_left = (left, top)
        bottom_right = (right, bottom)

        p0 = (int(left + dx), bottom)
        p1 = (left, int(bottom - (p0[1]-hor_y)/(p0[0]- hor_x)*dx))
        p2 = (p0[0], int(top + (top - hor_y)/(left-hor_x)*dx))
        p3 = (right, p2[1])
        p5 = (int(bottom_right[0] - (bottom_right[0] - hor_x) / (bottom_right[1] - hor_y) * (bottom - p1[1])), p1[1])
        p4 = (p5[0], top)

        back_box = (p2, bottom_right)
        front_box = (top_left, p5)


        # Front Box

        color = self.drawing_color
        thick = 2
        #cv2.rectangle(image, front_box[0], front_box[1], color, thickness=thick)
        cv2.rectangle(image, back_box[0], back_box[1], color, thickness=thick)

        cv2.line(image, top_left, p4, color, thickness=thick)
        cv2.line(image, top_left, p1, color, thickness=thick)
        cv2.line(image, top_left, p2, color, thickness=thick)
        cv2.line(image, p1, p0, color, thickness=thick)
        cv2.line(image, p4, p3, color, thickness=thick)
        #cv2.line(image, p5, bottom_right, color, thickness=thick)

        #b = self.car_box()

        # cv2.rectangle(image, b[0], b[1], (255,0,0), thickness=1)

	# draw draws the car as a box. This one is really used.
	#
	#	also writes the horizontal distance

    def draw(self, image, font):

        thick = 2

        if self.frames_seen <= 7 or self.frames_unseen > self.frames_seen:
            return

        if self.position[0] < 150:
            return

        if self.frames_seen > 2:
            thick = 6

        if self.frames_unseen > 1:
            thick = 2

        b = self.car_box()
        color = self.drawing_color
        color = (0, 0, 255)
        cv2.rectangle(image, b[0], b[1], color, thickness=thick)

        line_1 = "{}".format(self.name)
        line_2 = "{:0.2f} m".format(self.hor_distance())
        #cv2.putText(image, line_1, (int(self.position[0]-20), int(self.position[1]-10)), font, 1.0, (0, 0, 255), 2)
        cv2.putText(image, line_2, (int(self.position[0]-20), int(self.position[1])), font, 1.0, (0, 0, 255), 2)

	# Return car info in a string
	
    def car_info(self):

        return "Color {} Position {} Size {} Speed {} unseen {}".format(self.color, self.position, self.size, self.speed, self.frames_unseen)

    #Returns True if the a_car is nearer than self
    def isOcludedBy(self, a_car):

        # Oclussion menas there is intersection

        my_bbox = self.car_box()
        a_bbox = a_car.car_box()

        if not intersect(my_bbox, a_bbox):
            return False

        # OK, Now who is nearer?
        # Supose Horizon at half height in the window (camera just horizontal)
        #
        #   then (x-w/2)/(y-h/2) is the tg of the angle. Bigger angle means more to the right
        #
        #
        #   so a ocluddes us if  tg(us) > tg(a)



        return  a_car.hor_distance < self.hor_distance

    ## Returns distance to the right or left of a car

    def hor_distance(self):
        x = (self.position[0] - hor_x)
        y = (self.position[1]-hor_y+self.size[1]/2)
        return x / y * hor_y * 2 * 3.7 / 900 # 3.7/900 is scaling to meters

### Car_Collection is a collecton of cars


class Car_Collection:
    def __init__(self):

        # Car list
        self.cars = []
        self.car_number = 0
        self.colors = [(255,0,0),  (0,0,255), (255,255,0), (255,0,255), (0, 255,255)]
	
	# Add a car to the collection

    def add_car(self, car):
        self.car_number += 1
        car.name = self.car_number
        car.drawing_color = self.colors[self.car_number % len(self.colors)]
        car.sigma2 =(car.size[0], car.size[1])

        self.cars.append(car)
        
    # Tries to find cars compatible with the a_car measure

    def find_car(self, a_car, delta):

        results = []

        for car in self.cars:
            if car.distance(a_car.position) < delta*(car.frames_unseen+1):
                results.append(car)

            elif included(car.car_box(), a_car.car_box()):
                results.append(car)

            elif intersect(car.car_box(), a_car.car_box()): # No ho tinc tan clar
                results.append(car)

        return results
        
    # Executed for every frame to remove spuripus cars and other maintenances

    def prepare_for_update(self):

        new_cars = []
        for car in self.cars:
            if car.frames_unseen > car.frames_seen or car.frames_unseen > 7:
                # Remove car
                print("Removing car {}".format(car.name))
            else:
                car.frames_unseen += 1  # If seen will be set to 0
                new_cars.append(car)

            if car.speed is not None and False:
                car.position[0] += car.speed[0]
                car.position[1] += car.speed[1]

        self.cars = new_cars

	# Process a new observation

    def new_observation(self, car):

        # Look if there is a car that exists

        found_cars = self.find_car(car, 40)

        ## No cars, add it

        if len(found_cars) == 0:
            self.add_car(car)
            print("Added car", car.car_info())

        ## Found a car
        elif len(found_cars) == 1:
            the_car = found_cars[0]
            the_car.update_data(car)
            d = color_difference(the_car.color, car.color)
            print("Car {} with color difference {}".format(the_car.name, d))

        else:   # More than one car, try to select one

            the_car = None
            color_diff = 999999999999999
            area = 0

            for a_car in found_cars:
                if a_car.area() > area:
                    d = color_difference(a_car.color, car.color)
                    the_car = a_car
                    color_diff = d
                    area = a_car.area()

                    #if d < color_diff:
                    #the_car = a_car
                    #color_diff = d

            # great, I have a car
            the_car.update_data(car)
            print("Selected car {} with color difference {}".format(the_car.name, color_diff))

	# Draw the cars
	
    def draw_cars(self, image, font):
        for car in self.cars:
            if car.frames_seen > 5 and car.frames_unseen < car.frames_seen:
                car.draw(image, font)

            #car.draw(image, font)
        return image


### Main recognising function. 
#
#	Returns a list of posible cars, a heatmap and all the boxes related.
#
#	n is a parameter used to label false positives images


def recognize_cars(classifier, model, scaler, image,
                   false_positives_dir=None,
                   n=0,
                   threshold=3,
                   scales=[1],
                   ystart=400,
                   ystop=700,
                   input_boxes=None,
                   filter=False):

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    all_boxes = []

    if input_boxes is None:
        for i_scale in scales:

            if classifier == 'cnn':
                box_list = find_cars_boxes_cnn(image, model,
                                                ystart=ystart, #400
                                                ystop=ystop,
                                                scale=i_scale, #np.power(1.05, float(i_scale)),
                                                color_space='YUV', filter=filter)
            else:
                box_list = find_cars_boxes_svm(image, model, scaler,
                                                ystart=ystart, #400
                                                ystop=ystop,
                                                scale=i_scale, #np.power(1.05, float(i_scale)),
                                                color_space='YUV', filter=filter)


            all_boxes += box_list
            add_heat(heatmap, box_list)
    else:

        all_boxes += input_boxes
        add_heat(heatmap, input_boxes)

    car_boxes = find_car(heatmap,  threshold=threshold)

    # Try to get a list of cars from positions

    cars = []   # List of Car objects
    # Get car image
    for car_box in car_boxes:
        car_image = bbox_subimage(image, car_box)
        color = average_color(car_image)
        position = bbox_center(car_box)
        size = bbox_size(car_box)
        car = Car(position, size, color, image)
        cars.append(car)


    if false_positives_dir is not None:
        for box in all_boxes:
            isnotcar = True
            for car_box in car_boxes:
                if intersect(box, car_box): # Is a car,
                    isnotcar = False
                    break

            if isnotcar:
                n += 1
                cv2.imwrite(false_positives_dir+"/"+str(n)+".png", image[box[0][1]:box[1][1], box[0][0]:box[1][0]])


    return cars, car_boxes, heatmap, all_boxes, n



## Process folder
#
#	For every file in filenames try to recognize the car
#
def process_folder(classifier, filenames, model, scaler, false_positives_dir=None, log_folder=None, interactive=False):

    n = 0

    for image_name in filenames:
        print("Recognizing cars in " + image_name)
        img = cv2.imread(image_name)


        cars, car_boxes, heatmap, boxes, n = recognize_cars(classifier, model, scaler, img,
                                                 false_positives_dir=false_positives_dir,
                                                 threshold=17,  #8
                                                 n=n,
                                                 scales=[1,  1.5], filter=False)

        draw_img = draw_boxes(img, boxes, color=(255,0,0), thick=2)
        #draw_img = draw_boxes(draw_img, tight_boxes, color=(0, 255, 0))
        draw_img = draw_boxes(draw_img, car_boxes)


        # Convert heatmap to color


        max_h = np.max(heatmap)


        #color_heatmap = np.stack((np.zeros_like(heatmap),heatmap/max_h*128, heatmap/max_h*255), axis=2).astype(np.uint8)
        color_heatmap = np.stack((heatmap/max_h*15, np.ones_like(heatmap)*255, heatmap / max_h * 255), axis=2).astype(np.uint8)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_HSV2BGR)

        draw_heatmap = draw_boxes(color_heatmap, car_boxes, color=(0, 255, 0), thick=2)

        if log_folder is not None:
            name = image_name.split("/")[-1]
            cv2.imwrite(log_folder+"/"+name, draw_img)

        print(image_name, cars)

        if interactive:
            fig = plt.figure(figsize=(16,10))
            plt.subplot(221)
            plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            plt.title('Example Car Image')
            plt.subplot(222)
            #plt.imshow(cv2.cvtColor(draw_heatmap, cv2.COLOR_BGR2RGB))

            plt.imshow(heatmap, cmap='hot')

            plt.title('Heatmap')
            fig.tight_layout()
            plt.show()

        '''
        cv2.imshow("heatmap", heatmap)

        cv2.imshow("Output", draw_img)

        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        else:
            continue

        '''


## Process video
#
# filename is the name of the folder
# first is first frame to process
# last is last frame to process
# log_folder if not None is a folder where to store each original frame
# output_video if not None is the pathname where the edited viceo will be written


def process_video(classifier, model, scaler, filename,
                  first=1,
                  last=100000,
                  log_folder=None,
                  output_video=None,
                  false_positives_dir = None,
                  interactive=False,
                  dump_boxes_file=None,
                  load_boxes=False):


    font = cv2.FONT_HERSHEY_PLAIN
    if interactive:
        win = cv2.namedWindow('Video')

    old_filtereds = []

    cap = cv2.VideoCapture(filename)  # Careful images are BGR

    frame = 0
    ret = True
    n = 0

    history_boxes = []

    next_box = 0

    dump_boxes = dump_boxes_file is not None and not load_boxes


    if load_boxes and dump_boxes_file is not None:
        with open(dump_boxes_file, "rb") as f:
            history_boxes = pickle.load(f)


    car_collection = Car_Collection()

    if output_video is not None:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(output_video, fourcc, 24,
                                 (1280, 720), True)

    print("Jumping over first {} frames".format(first))
    while ret and frame < first:  # Skip over the non wanted frames
        ret, image = cap.read()
        frame = frame + 1
        if load_boxes:              # next_box apunta al seguent frame
            while history_boxes[next_box][0] < frame:
                next_box += 1



    print("Beginning car localization")

    while ret and frame <= last:
        # Display data

        if load_boxes:
            box_list = []
            while next_box < len(history_boxes) and history_boxes[next_box][0] == frame:
                box_list = history_boxes[next_box][1]
                next_box += 1
        else:
            box_list = None

        cars, car_boxes, heatmap, boxes, n = recognize_cars(classifier, model, scaler, image,
                            false_positives_dir=false_positives_dir,
                            threshold=8,    # 8 per NN
                            n=n,
                            scales=[1, 1.5],
                            input_boxes=box_list)

        if dump_boxes:
            history_boxes.append((frame, boxes))

        print("Frame {} found {} cars".format(frame, len(cars)))

        car_collection.prepare_for_update()     # That increases unseen_frames for all cars. Will be reset when updated
        for car in cars:
            print("    ", car.car_info())

        for car in cars:
            car_collection.new_observation(car)

        #edited_frame = draw_boxes(image, car_boxes, color=(0, 255, 0), thick=1)
        edited_frame = draw_boxes(image, [], color=(0, 255, 0), thick=1)

        car_collection.draw_cars(edited_frame, font)


        if log_folder is not None:
            f = "{}/{}.jpg".format(log_folder, frame)
            cv2.imwrite(f, image)


        # Write frame to video output
        if output_video is not None:
            writer.write(edited_frame)

        ret, image = cap.read()
        frame = frame + 1

        if interactive:
            cv2.imshow('Video', edited_frame)  # was out_img - may be warped to analysis

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            else:
                continue

        if ret % 10 == 9 and dump_boxes_file is not None and dump_boxes:
                with open("t_" + dump_boxes_file, "wb") as f:
                    pickle.dump(history_boxes, f)

    if output_video is not None:
        writer.release()

    if interactive:
        cv2.destroyAllWindows()

    # Now save hustory to a file

    if dump_boxes_file is not None and dump_boxes:
        with open(dump_boxes_file, "wb") as f:
            pickle.dump(history_boxes, f)

    for car in car_collection.cars:
        print(car.car_info())


### Main program


if len(sys.argv) < 2:
    print("python find_cars <filename> [-video] [-o filename] [-c<cnn|svm>] [-m <model>] [-sc <scaler>] [-first <first frame>] [-last <last_frame>] [-fp <false positives dir>] [-log <log_image_dir] [-d dump file] [-l boxes file]")
    exit(1)

mode = 'photo'
classifier = 'cnn'
model_name = "./models/new_model.hf5"
scaler_name = "scaler.pkl"
out_filename = None
false_positives_dir = None
log_dir = None

first_frame = 1
last_frame = 10000

model = None
scaler = None

interactive = False
dump_boxes_file = None
dump_boxes = False
load_boxes = False

max_arg = len(sys.argv) - 1

for i, v in enumerate(sys.argv):
    if i == 1:
        input_filename = sys.argv[1]

    elif v == '-o':
        if i < max_arg:
            out_filename = sys.argv[i+1]

    elif v == '-c':
        if i < max_arg:
            classifier = sys.argv[i+1]

    elif v == '-m':
        if i < max_arg:
            model_name = sys.argv[i+1]

    elif v == '-sc':
        if i < max_arg:
            scaler_name = sys.argv[i+1]

    elif v == '-fp':
        if i < max_arg:
            false_positives_dir = sys.argv[i+1]

    elif v == '-log':
        if i < max_arg:
            log_dir = sys.argv[i + 1]

    elif v == '-first':
        if i < max_arg:
            first_frame = int(sys.argv[i+1])

    elif v == '-last':
        if i < max_arg:
            last_frame = int(sys.argv[i + 1])
    elif v == '-video':
        mode = 'video'

    elif v == '-interactive':
        interactive = True

    elif v == '-d':
        if i < max_arg:
            dump_boxes_file = sys.argv[i + 1]
            dump_boxes = True
            load_boxes = False

    elif v == '-l':
        if i < max_arg:
            dump_boxes_file = sys.argv[i + 1]
            load_boxes = True
            dump_boxes = False


print("Processing {} as {} into {}".format(input_filename, mode, out_filename))
if mode == 'video':
    print("From frame {} to frame {}".format(first_frame, last_frame))

print("Using {} classifier with model {}".format(classifier, model_name))

if false_positives_dir is not None:
    print("False positives will be writen to {}".format(false_positives_dir))

if log_dir is not None:
    print("Images will be written to {}".format(log_dir))

print("---------------------------------")

print("Loading model", model_name, "...")

if classifier == 'cnn':
    model = load_model(model_name)
else:
    with open(model_name, "rb") as f:
        model = pickle.load(f)

    with open(scaler_name, "rb") as f:
        scaler = pickle.load(f)


print ("Model Loaded")

if mode == 'photo':
    filenames = glob(input_filename + "/*.jpg")
    process_folder(classifier, filenames, model, scaler,
                   log_folder=log_dir,
                   false_positives_dir=false_positives_dir,
                   interactive=interactive)

elif mode == 'video':
    process_video(classifier, model, scaler, input_filename,
                  first=first_frame,
                  last=last_frame,
                  log_folder=log_dir,
                  false_positives_dir=false_positives_dir,
                  output_video=out_filename,
                  interactive=interactive,
                  dump_boxes_file=dump_boxes_file,
                  load_boxes=load_boxes)
