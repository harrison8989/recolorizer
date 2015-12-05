#!/usr/bin/env python
import os
import code

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

YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T

def retrieveYUV(img):
    return np.dot(img, YUV_FROM_RGB)

# Given an image, return
#    - An array of subsquares
#    - An array containing the average U values for the subsquares
#    - An array containing the average V values for the subsquares
def generateSubsquares(path):
    img, segments = segment_image(path)
    yuv = retrieveYUV(img)
    n_segments = segments.max() + 1
    # code.InteractiveConsole(locals=locals()).interact()

    # Compute the centroids/average U and V of each of the superpixels
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    U = np.zeros(n_segments)
    V = np.zeros(n_segments)
    for (i,j), value in np.ndenumerate(segments):
        point_count[value] += 1
        centroids[value][0] += i
        centroids[value][1] += j
        U[value] += yuv[i][j][1]
        V[value] += yuv[i][j][2]

    for k in range(n_segments):
        centroids[k] /= point_count[k]
        U[k] /= point_count[k]
        V[k] /= point_count[k]

    # Generate the array of squares
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

    return subsquares, U, V

if __name__ == '__main__':
    u_svr = SVR(C=1.0, epsilon=.1)
    v_svr = SVR(C=1.0, epsilon=.1)
    for root, dirs, files in os.walk('data/flickr/'):
        X = np.array([]).reshape(0, SQUARE_SIZE * SQUARE_SIZE)
        U_L = np.array([])
        V_L = np.array([])

        numIters = 0

        for file in files:
            path = os.path.join(root, file)
            if not path.endswith(".jpg"):
                continue

            print "Training on", path, "..."
            subsquares, U, V = generateSubsquares(path)

            X = np.concatenate((X, subsquares), axis=0)
            U_L = np.concatenate((U_L, U), axis=0)
            V_L = np.concatenate((V_L, V), axis=0)

            numIters += 1
            if numIters > 100:
                break

        u_svr.fit(X, U_L)
        v_svr.fit(X, V_L)
        util.mkdirp('models/')
        joblib.dump(u_svr, 'models/u_svr.model')
        joblib.dump(v_svr, 'models/v_svr.model')
