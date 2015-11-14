#!/user/bin/env python
import argparse
import code
import os

from sklearn.svm import SVR
from skimage.segmentation import slic, mark_boundaries
from skimage.data import imread
from skimage.util import img_as_float
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np

from constants import *
from segment_images import segment_image
import util

RGB_FROM_YUV = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                       [-0.14713, -0.28886, 0.436],
                                       [0.615, -0.51499, -0.10001]]).T)


def retrieveRGB(img):
    return np.dot(img, RGB_FROM_YUV)

# Given a black and white image, predict its chrominance (U and V values in YUV space)
def predict_image(u_svr, v_svr, path):
    img, segments = segment_image(path)
    n_segments = segments.max()

    # Construct the centroids of the image
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    for (i,j), value in np.ndenumerate(segments):
        point_count[value - 1] += 1
        centroids[value - 1][0] += i
        centroids[value - 1][1] += j
    for k in range(n_segments):
        centroids[k] /= point_count[k]

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
                subsquares[k][i*SQUARE_SIZE + j] = img[i + top][j + left][0]

    # Predict using SVR
    predicted_u = np.zeros(n_segments)
    predicted_v = np.zeros(n_segments)
    for k in range(n_segments):
        predicted_u[k] = u_svr.predict(subsquares[k])
        predicted_v[k] = u_svr.predict(subsquares[k])

    # Reconstruct images
    for (i,j), value in np.ndenumerate(segments):
        img[i][j][1] = predicted_u[value-1]
        img[i][j][2] = predicted_v[value-1]

    rgb = retrieveRGB(img)
    plt.imshow(rgb)
    plt.show()

if __name__ == '__main__':
    u_svr = joblib.load('models/u_svr.model')
    v_svr = joblib.load('models/v_svr.model')

    parser = argparse.ArgumentParser(description='Test the SVM on an image.')
    parser.add_argument('file', metavar='file_name', help='The image to be tested')
    args = parser.parse_args()
    predict_image(u_svr, v_svr, args.file)
