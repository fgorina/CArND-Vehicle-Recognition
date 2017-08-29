import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
### Utility functions
#

## BBoxes utility

### subimage returns a subimage selected by the bbox

def bbox_subimage(img, bbox):
    return img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

## Returns ceter as (x, y)

def bbox_center(bbox):
    return (int((bbox[0][0]+bbox[1][0])/2), int((bbox[0][1]+bbox[1][1])/2))

## returns size as (width, height)
def bbox_size(bbox):
    return (bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1])

## Moves bbox dx, dy
#
# returns new bbox

def bbox_move(bbox, dx, dy):
    return ((bbox[0][0]+dx, bbox[0][1]+dy), (bbox[1][0]+dx, bbox[1][1]+dy))

## increase returns a new bbox with size 2dx and 2dy more centered at same point
def bbox_increase(bbox, dx, dy):
    return ((bbox[0][0] - dx, bbox[0][1] - dy), (bbox[1][0] + dx, bbox[1][1] + dy))

## average color
#
# computes average color of image
def average_color(image):
    return np.average(image, axis=(0, 1))

## returns averaged center of bbox. img just one channel
#
# retuns (x, y) of center

def bbox_averaged_center(img, bbox):

    sub_image = bbox_subimage(img, bbox) # Extract the subimage

    x = range(0, sub_image.shape[1])
    y = range(0, sub_image.shape[0])

    (X, Y) = np.meshgrid(x, y)

    x_coord = (X * sub_image).sum() / sub_image.sum().astype("float")
    y_coord = (Y * sub_image).sum() / sub_image.sum().astype("float")

    return (int(x_coord + bbox[0][0]), int(y_coord + bbox[0][1]))

### New bbox correcponding to center and size
#
def bbox_new(center, size):

    return (
        (int(center[0]-size[0]/2), int(center[1] - size[1]/2)),
        (int(center[0]+size[0]/2), int(center[1] + size[1]/2))
        )

### Color Difference computes a euclidian color distance

def color_difference(c1, c2):
    return np.sqrt(np.sum((c1-c2)*(c1-c2)))

### Distance in pixels from 2 points

def distance(p1, p2):
    d1 = p1[0]-p2[0]
    d2 = p1[1]-p2[1]
    return np.sqrt(d1*d1+d2*d2)


## Returns converted image. a conv put only colorspace
def convert_color(img, conv='YCrCb'):

    if conv != 'BGR':
        conversion = "cv2.COLOR_BGR2" + conv
        conv_code = eval(conversion)
        return cv2.cvtColor(img, conv_code)
    else:
        return np.copy(img)
        
## Gets HOG features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

## Returns bin spatial features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

## Returns a color histogram

def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    norm_hist_features = hist_features.astype(np.float32)/np.max(hist_features)
    # Return the individual histograms, bin_centers and feature vector
    return norm_hist_features


## Extract features from an BGR image after converting colorspace
def extract_img_features(img_file, color_space='YUV',
                         spatial_size=(32, 32),
                         hist_bins=32,
                         orient=9,
                         pix_per_cell=8,
                         cell_per_block=2,
                         spatial_feat=True,
                         hist_feat=True,
                         hog_feat=True,
                         hog_channel='ALL'):

    img_features = []
    img = cv2.imread(img_file)
    img = cv2.resize(img, (64,64))

    feature_image = convert_color(img, color_space).astype(np.float32) / 255

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(np.array(hog_features, dtype=np.float32))

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Process data for NN Is not used now

def process(image, filter=False):

    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img = (img - 128) / 255
    return img



# Define a single function that can extract features using hog sub-sampling and make predictions
# The proposed function

def find_cars_boxes_svm(img, svc, X_scaler, ystart=None, ystop=None, scale=1,
              orient=9,
              pix_per_cell=8,
              cell_per_block=2,
              spatial_size=(32,32),
              hist_bins=32,
              color_space='YUV', filter=False):


    img = img.astype(np.float32) / 255

    if ystart is None:
        ystart = int(img.shape[0]/2)

    if ystop is None:
        ystop = int(img.shape[0])

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)

    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    box_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = X_scaler.transform(
            unscaled_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            #unscaled_features = np.hstack((spatial_features,  hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(unscaled_features)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                box_list.append(((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #             (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return box_list


## Define a single function that can extract features compatible with the svm find_car_boxes function
#
# Scans the image with the same sliding window algorith but applies a CNN model

def find_cars_boxes_cnn(img, model, ystart=None, ystop=None, scale=1, color_space='YUV', filter=False):

    if ystart is None:
        ystart = int(img.shape[0]/2)

    if ystop is None:
        ystop = int(img.shape[0])

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    ctrans_tosearch = (ctrans_tosearch - 128) / 255

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    pixels_per_step = 16

    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps =  (ctrans_tosearch.shape[1]-window+pixels_per_step)//pixels_per_step
    nysteps =  (ctrans_tosearch.shape[0]-window+pixels_per_step)//pixels_per_step

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

            all_box_list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

            subimg = ctrans_tosearch[ypos:ypos + window, xpos:xpos + window]
            image_list.append(subimg)

    x_data = np.array(image_list)

    x_pred = model.predict(x_data)

    for result, box in zip(x_pred, all_box_list):
        value = np.argmax(result)
        if value == 0:
             box_list.append(box)

    return box_list

### Hetmap functions
#
# finally convert collections of boxes to single car boxes
# returns list of cars/boxes

def add_heat(heatmap, bbox_list):

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

def find_car(heat, threshold=1):
    # Apply a threshold

    heatmap = np.copy(heat)
    heatmap[heatmap <= threshold] = 0

    # Label the cars. Label marks connected areas
    labels = label(heatmap)

    car_list = []

    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        center = bbox_averaged_center(heatmap, bbox)
        size = bbox_size(bbox)

        if size[0] > 30 and size[1] > 30:
            bbox = bbox_new(center,size)
            car_list.append(bbox)

    return car_list

## draw_boxes draws a box list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn

    for b in bboxes:
        tl = b[0]
        br = b[1]
        cv2.rectangle(draw_img, b[0], b[1], color, thickness=thick)

    return draw_img # Change this line to return image copy with boxes

## draw_cars draws a car list

def draw_cars(img, cars, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn

    for car in cars:
        b = car.car_box()
        cv2.rectangle(draw_img, b[0], b[1], color, thickness=thick)

    return draw_img # Change this line to return image copy with boxes

## Return true if 2 boxes intersect

def intersect(box1, box2):

    xmin_1 = min(box1[0][0], box1[1][0])
    xmax_1 = max(box1[0][0], box1[1][0])
    ymin_1 = min(box1[0][1], box1[1][1])
    ymax_1 = max(box1[0][1], box1[1][1])

    xmin_2 = min(box2[0][0], box2[1][0])
    xmax_2 = max(box2[0][0], box2[1][0])
    ymin_2 = min(box2[0][1], box2[1][1])
    ymax_2 = max(box2[0][1], box2[1][1])

    if xmin_1 > xmax_2 or xmax_1 < xmin_2 or ymin_1 > ymax_2 or ymax_1 < ymin_2:
        return False
    else:
        return True

## Return true if one box includes the other

def included(box1, box2):

    xmin_1 = min(box1[0][0], box1[1][0])
    xmax_1 = max(box1[0][0], box1[1][0])
    ymin_1 = min(box1[0][1], box1[1][1])
    ymax_1 = max(box1[0][1], box1[1][1])

    xmin_2 = min(box2[0][0], box2[1][0])
    xmax_2 = max(box2[0][0], box2[1][0])
    ymin_2 = min(box2[0][1], box2[1][1])
    ymax_2 = max(box2[0][1], box2[1][1])

    # No possible if they don't intersect

    if xmin_1 > xmax_2 or xmax_1 < xmin_2 or ymin_1 > ymax_2 or ymax_1 < ymin_2:
        return False

    if xmin_1 <= xmin_2:
        if xmax_1 >= xmax_2 and ymin_1 <= ymin_2 and ymax_1 >= ymax_2:
            return True

    elif xmin_1 <= xmin_2:
        if xmax_1 <= xmax_2 and ymin_1 >= ymin_2 and ymax_1 <= ymax_2:
            return True

    return False
