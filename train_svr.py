#!/user/bin/env python
import os

from sklearn.svm import SVR
from skimage.segmentation import slic, mark_boundaries
from skimage.data import imread
from skimage.util import img_as_float

import matplotlib.pyplot as plt
import numpy as np

YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T

def retrieveYUV(img):
    return np.dot(img, YUV_FROM_RGB)

for root, dirs, files in os.walk('data/flickr/'):
    for file in files:
        path = os.path.join(root, file)
        img = img_as_float(imread(path))

        yuv = retrieveYUV(img)
        segments = slic(img, n_segments=250, compactness=10, sigma=1)
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax[0].imshow(mark_boundaries(img, segments_fz))
        plt.show()
        quit()
