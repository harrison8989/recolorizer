#!/usr/bin/env python
import os
from atomicfile import AtomicFile

from skimage.segmentation import slic
from skimage.data import imread
from skimage.util import img_as_float

import numpy as np
from constants import *


def segment_image(path):
    img = img_as_float(imread(path))
    segment_file = path + "." + str(N_SEGMENTS) + ".segments"
    if os.path.isfile(segment_file):
        return img, np.load(segment_file)

    print "Segmenting ", path
    segments = slic(img, n_segments=N_SEGMENTS, compactness=10, sigma=1)

    with AtomicFile(segment_file, 'wb') as fd:
        np.save(fd, segments)

    return img, segments

# When run as a script, segment all of the files in the data directory.
if __name__ == '__main__':
    for root, subdirs, files in os.walk('data'):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith(".jpg"):
                segment_image(path)
