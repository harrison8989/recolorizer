#!/usr/bin/env python
import argparse
import code
import os

from sklearn.svm import SVR
from skimage.segmentation import slic, mark_boundaries
from skimage.data import imread
from skimage.io import imsave
from skimage.util import img_as_float
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np

from constants import *
from segment_images import segment_image
import util

# TODO: Put these in constants folder
RGB_FROM_YUV = np.array([[1, 0, 1.13983],
                         [1, -0.39465, -.58060],
                         [1, 2.03211, 0]]).T
YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T
U_MAX = 0.436
V_MAX = 0.615

def clamp(val, low, high):
    return np.maximum(np.minimum(val, high), low)

def clampU(val):
    return clamp(val, -U_MAX, U_MAX)

def clampV(val):
    return clamp(val, -V_MAX, U_MAX)

def retrieveRGB(img):
    return clamp(np.dot(img, RGB_FROM_YUV), 0, 1)

def retrieveYUV(img):
    return np.dot(img, YUV_FROM_RGB)

# Generates the adjacency list for each of the segments in the image.
def generate_adjacencies(segments, n_segments, img, subsquares):
    adjacency_list = []
    for i in range(n_segments):
        adjacency_list.append(set())
    for (i,j), value in np.ndenumerate(segments):
        # Check vertical adjacency
        if i < img.shape[0] - 1:
            newValue = segments[i + 1][j]
            if value != newValue and np.linalg.norm(subsquares[value] - subsquares[newValue]) < THRESHOLD:
                adjacency_list[value].add(newValue)
                adjacency_list[newValue].add(value)

        # Check horizontal adjacency
        if j < img.shape[1] - 1:
            newValue = segments[i][j + 1]
            if value != newValue and np.linalg.norm(subsquares[value] - subsquares[newValue]) < THRESHOLD:
                adjacency_list[value].add(newValue)
                adjacency_list[newValue].add(value)

    return adjacency_list

# Given the prior observed_u and observed_v, which are generated using the SVR,
# represent the system as a Markov Random Field and optimize over it using
# Iterated Conditional Modes. Return the prediction of the hidden U and V values
# of the segments.
# For now, we assume that the U and V channels behave independently.
def apply_mrf(observed_u, observed_v, segments, n_segments, img, subsquares):
    hidden_u = np.copy(observed_u)  # Initialize hidden U and V to the observed
    hidden_v = np.copy(observed_v)

    adjacency_list = generate_adjacencies(segments, n_segments, img, subsquares)

    for iteration in range(ICM_ITERATIONS):
        new_u = np.zeros(n_segments)
        new_v = np.zeros(n_segments)

        for k in range(n_segments):

            u_potential = 100000
            v_potential = 100000
            u_min = -1
            v_min = -1

            # Compute conditional probability over all possibilities of U
            for u in np.arange(-U_MAX, U_MAX, .001):
                u_computed = (u - observed_u[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    u_computed += WEIGHT_DIFF * ((u - hidden_u[adjacency]) ** 2)
                if u_computed < u_potential:
                    u_potential = u_computed
                    u_min = u
            new_u[k] = u_min

            # Compute conditional probability over all possibilities of V
            for v in np.arange(-V_MAX, V_MAX, .001):
                v_computed = (v - observed_v[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    v_computed += WEIGHT_DIFF * ((v - hidden_v[adjacency]) ** 2)
                if v_computed < v_potential:
                    v_potential = v_computed
                    v_min = v
            new_v[k] = v_min

        u_diff = np.linalg.norm(hidden_u - new_u)
        v_diff = np.linalg.norm(hidden_v - new_v)
        hidden_u = new_u
        hidden_v = new_v
        if u_diff < ITER_EPSILON and v_diff < ITER_EPSILON:
            break

    return hidden_u, hidden_v

# Given an image, predict its chrominance (U and V values in YUV space)
def predict_image(u_svr, v_svr, path, verbose, output_file = None):
    img, segments = segment_image(path)
    yuv = retrieveYUV(img)   # Use first component of yuv to obtain black and white
    n_segments = segments.max() + 1

    # Construct the centroids of the image
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    luminance = np.zeros(n_segments)

    for (i,j), value in np.ndenumerate(segments):
        point_count[value] += 1
        centroids[value][0] += i
        centroids[value][1] += j
        luminance[value] += yuv[i][j][0]

    for k in range(n_segments):
        centroids[k] /= point_count[k]
        luminance[k] /= point_count[k]

    # Generate the subsquares
    subsquares = np.zeros((n_segments, SQUARE_SIZE * SQUARE_SIZE))
    for k in range(n_segments):
        # Check that the square lies completely within the image
        top = max(int(centroids[k][0]), 0)
        if top + SQUARE_SIZE >= img.shape[0]:
            top = img.shape[0] - 1 - SQUARE_SIZE
        left = max(int(centroids[k][1]), 0)
        if left + SQUARE_SIZE >= img.shape[1]:
            left = img.shape[1] - 1 - SQUARE_SIZE
        for i in range(0, SQUARE_SIZE):
            for j in range(0, SQUARE_SIZE):
                subsquares[k][i*SQUARE_SIZE + j] = yuv[i + top][j + left][0]
        subsquares[k] = np.fft.fft2(subsquares[k].reshape(SQUARE_SIZE, SQUARE_SIZE)).reshape(SQUARE_SIZE * SQUARE_SIZE)

    # Predict using SVR
    predicted_u = clampU(u_svr.predict(subsquares)*2)
    predicted_v = clampV(v_svr.predict(subsquares)*2)

    # Apply MRF to smooth out colorings
    predicted_u, predicted_v = apply_mrf(predicted_u, predicted_v, segments, n_segments, img, subsquares)

    # Reconstruct images
    for (i,j), value in np.ndenumerate(segments):
        yuv[i][j][1] = predicted_u[value]
        yuv[i][j][2] = predicted_v[value]
    rgb = retrieveRGB(yuv)

    # Compute the norm error. Note that it will be wildly incorrect if the img is b/w.
    error = 1000 * np.linalg.norm(rgb - img) / (img.shape[0] * img.shape[1])

    if verbose:
        print 'Norm error:', error
        # Draw the actual figure
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(rgb)
        if output_file:
            imsave(output_file, rgb)
        plt.show()

    return error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the SVR on an image.')
    parser.add_argument('file', metavar='file_name', help='The image to be tested. If -a or -c is selected, the script will instead test on the testing set.')
    parser.add_argument('-a', help='Specify training the different models', action='store_true')
    parser.add_argument('-c', help='Specify training the model over different variables for ICM.', action='store_true')
    parser.add_argument('-f', metavar='output_file', help='Output file for model', default='svr.model')
    parser.add_argument('-s', metavar='save_location', help='Location to save the file', default='out.png')
    args = parser.parse_args()

    if args.a:   # Test each of the models on the given data set.
        u_svrs = []
        v_svrs = []
        print 'Loading models:'
        for model in range(len(C_LIST) * len(EPSILON_LIST)):
            u_svrs.append(joblib.load('models/u_svr' + str(model) + '.model'))
            v_svrs.append(joblib.load('models/v_svr' + str(model) + '.model'))
        print 'Finished loading models.\n'

        for model in range(len(C_LIST) * len(EPSILON_LIST)):
            print 'Running predictions for model', model
            total_error = 0
            num_files = 0

            for root, dirs, files in os.walk(args.file):
                for file in files:
                    path = os.path.join(root, file)
                    if path.endswith('.jpg'):
                        total_error += predict_image(u_svrs[model], v_svrs[model], path, False)
                        num_files += 1
            print 'Total error for model', model, ':', total_error / num_files

    if args.c:   # Test each of the models on the given data set, modifying the ICM constants.
        u_svr = joblib.load('models/u_svr.model')
        v_svr = joblib.load('models/v_svr.model')
        for weight_diff in WEIGHT_DIFF_LIST:
            print 'Running predictions for weight difference:', weight_diff
            WEIGHT_DIFF = weight_diff
            total_error = 0
            num_files = 0

            for root, dirs, files in os.walk(args.file):
                for file in files:
                    path = os.path.join(root, file)
                    if path.endswith('.jpg'):
                        total_error += predict_image(u_svr, v_svr, path, False)
                        num_files += 1
            print 'Total error for weight diff', weight_diff, ':', total_error / num_files
        WEIGHT_DIFF = 2

        for threshold in THRESHOLD_LIST:
            print 'Running predictions for threshold:', threshold
            THRESHOLD = threshold
            total_error = 0
            num_files = 0

            for root, dirs, files in os.walk(args.file):
                for file in files:
                    path = os.path.join(root, file)
                    if path.endswith('.jpg'):
                        total_error += predict_image(u_svr, v_svr, path, False)
                        num_files += 1
            print 'Total error for threshold', threshold, ':', total_error / num_files


    else:        # Test on the single image
        u_svr = joblib.load('models/u_' + args.f)
        v_svr = joblib.load('models/v_' + args.f)
        predict_image(u_svr, v_svr, args.file, True, args.s)
