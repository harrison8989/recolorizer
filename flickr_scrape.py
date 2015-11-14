#!/usr/bin/env python
import requests
import string
import os.path
from atomicfile import AtomicFile

# URL template for a Flickr image.
photo_url = string.Template("https://farm${farmid}.staticflickr.com/${serverid}/${id}_${secret}.jpg")
# Page counter.
page = 1

while True:
    # Search for images with the desired tags.
    r = requests.get('https://api.flickr.com/services/rest/', params={
        'method': 'flickr.photos.search',
        'format': 'json',
        'nojsoncallback': 1,
        'api_key': '0b34944b5b61b43ec4fb3dc6389377a6',
        'tags': 'yellowstone,landscape',
        'tag_mode': 'all',
        'page': page,
    })

    json = r.json()
    for photo in json['photos']['photo']:
        file = "data/flickr/" + photo['id'] + '.jpg'
        url = photo_url.substitute(farmid=photo['farm'], serverid=photo['server'], id=photo['id'], secret=photo['secret'])

        # Only try to create the file if it doesn't exist already.
        if not os.path.isfile(file):
            print "Downloading " + url + " to " + file + "..."
            photo_req = requests.get(url)
            if photo_req.status_code != requests.codes.ok:
                print "Error requesting photo."
                break

            # Write the file atomically, so we don't have corrupted images.
            with AtomicFile(file, 'wb') as fd:
                for chunk in photo_req.iter_content(1024):
                    fd.write(chunk)
        else:
            print file + " already exists..."

    # Continue until we've exhausted all of the pages.
    if page < json['photos']['pages']:
        print "Querying page", page, "out of", json['photos']['pages']
        page += 1
