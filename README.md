# AI Image Recolorizer

![Imgur](https://i.imgur.com/HNee6oE.png)

[Research Paper](http://cs229.stanford.edu/proj2015/150_report.pdf)

## Setup

First, install dependencies via pip:

    pip install -r requirements.txt

We provide a script to automatically scrape images from Flickr. Run
the script to acquire a dataset:

    python flickr_scrape.py

Generate the SVR model by running

    python train_svr.py

Finally, test the SVR model on a single photo by running

    python test_svr.py <file name>
