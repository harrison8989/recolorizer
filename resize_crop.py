#!/usr/bin/env python
import os
from atomicfile import AtomicFile
from PIL import Image

# The desired size that all images in the data directory should
# be normalized to.
DESIRED_SIZE = (500, 500)

for root, subdirs, files in os.walk('data'):
    for file in files:
        path = os.path.join(root, file)
        print "Opening " + path
        im = Image.open(path)
        if im.size != DESIRED_SIZE:
            print "Cropping / resizing", path, " to ", DESIRED_SIZE
