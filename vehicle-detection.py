import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from pathlib import Path
import pickle

class par:
    ##Hyperparameters
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 15  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (8, 8)  # Spatial binning dimensions
    hist_bins = 64  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start = 400
    y_stop = 656  # Min and max in y to search in slide_window()
    scale = 1.5
    # Classifier
    svc = []
    X_scaler = []

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=False):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def train_classifier(sample_size=-1):
    # Read in cars and notcars
    notcars = glob.glob('non-vehicles/**/*.png')
    cars = glob.glob('vehicles/**/*.png')

    # Reduce the sample size to reduce training time
    cars = cars[0:sample_size:5]
    notcars = notcars[0:sample_size:5]

    car_features = extract_features(cars, color_space=par.color_space,
                                    spatial_size=par.spatial_size, hist_bins=par.hist_bins,
                                    orient=par.orient, pix_per_cell=par.pix_per_cell,
                                    cell_per_block=par.cell_per_block,
                                    hog_channel=par.hog_channel, spatial_feat=par.spatial_feat,
                                    hist_feat=par.hist_feat, hog_feat=par.hog_feat)
    notcar_features = extract_features(notcars, color_space=par.color_space,
                                       spatial_size=par.spatial_size, hist_bins=par.hist_bins,
                                       orient=par.orient, pix_per_cell=par.pix_per_cell,
                                       cell_per_block=par.cell_per_block,
                                       hog_channel=par.hog_channel,
                                       spatial_feat=par.spatial_feat,
                                       hist_feat=par.hist_feat, hog_feat=par.hog_feat)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    par.X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = par.X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.3, random_state=rand_state)

    print('Using:', par.orient, 'orientations', par.pix_per_cell,
          'pixels per cell and', par.cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    par.svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    par.svc.fit(X_train, y_train)
    t2 = time.time()
    print(t2 - t, 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(par.svc.score(X_test, y_test), 4))

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img):
    img = img.astype(np.float32) / 255

    bboxes = []

    img_tosearch = img[par.y_start:par.y_stop, :, :]
    if par.color_space != 'RGB':
        if par.color_space == 'HSV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif par.color_space == 'LUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif par.color_space == 'HLS':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif par.color_space == 'YUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif par.color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

    else: feature_image = np.copy(img_tosearch)

    if par.scale != 1:
        imshape = feature_image.shape
        feature_image = cv2.resize(feature_image, (np.int(imshape[1] / par.scale), np.int(imshape[0] / par.scale)))

    ch1 = feature_image[:, :, 0]
    ch2 = feature_image[:, :, 1]
    ch3 = feature_image[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // par.pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // par.pix_per_cell) - 1
    nfeat_per_block = par.orient * par.cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // par.pix_per_cell) - 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if par.hog_feat == True:
        if par.hog_channel == 'ALL':
            hog1 = get_hog_features(ch1, par.orient, par.pix_per_cell, par.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, par.orient, par.pix_per_cell, par.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, par.orient, par.pix_per_cell, par.cell_per_block, feature_vec=False)
        elif par.hog_channel == 1:
            hog1 = get_hog_features(ch1, par.orient, par.pix_per_cell, par.cell_per_block, feature_vec=False)
        elif par.hog_channel == 2:
            hog2 = get_hog_features(ch2, par.orient, par.pix_per_cell, par.cell_per_block, feature_vec=False)
        else:
            hog3 = get_hog_features(ch3, par.orient, par.pix_per_cell, par.cell_per_block, feature_vec=False)


    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            if par.hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            elif par.hog_channel == 1:
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = []
                hog_feat3 = []
            elif par.hog_channel == 2:
                hog_feat1 = []
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = []
            else:
                hog_feat1 = []
                hog_feat2 = []
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * par.pix_per_cell
            ytop = ypos * par.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(feature_image[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            if par.spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=par.spatial_size)
            else:
                spatial_features = []

            if par.hist_feat == True:
                hist_features = color_hist(subimg, nbins=par.hist_bins)
            else:
                hist_features = []

           # Scale features and make a prediction
            test_features = par.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = par.svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * par.scale)
                ytop_draw = np.int(ytop * par.scale)
                win_draw = np.int(window * par.scale)
                bboxes.append(((xbox_left, ytop_draw + par.y_start), (xbox_left + win_draw, ytop_draw + win_draw + par.y_start)))


    return bboxes


def pipeline(image):

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Plot all Windows
    hot_windows = find_cars(image)
    window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Plot heatmap
    #plt.imshow(heatmap)
    #plt.show()

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(np.copy(image), labels)

    # Final image with detected cars
    #plt.imshow(draw_image)
    #plt.show()

    return draw_image

# Main
train_classifier(sample_size=-1)

# Test Images
for i in range(1,6):
    image = mpimg.imread('test_images/test' + str(i) + '.jpg')
    pipeline(image)

# Video
#input = VideoFileClip("project_video.mp4")
#output = input.fl_image(pipeline)
#output.write_videofile('result.mp4', audio=False)

