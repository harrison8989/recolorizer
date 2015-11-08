import requests
import string
import os.path
import shutil

photo_url= string.Template("https://farm${farmid}.staticflickr.com/${serverid}/${id}_${secret}.jpg")

r = requests.get('https://api.flickr.com/services/rest/', params = {
    'method': 'flickr.photos.search',
    'format': 'json',
    'nojsoncallback': 1,
    'api_key': '0b34944b5b61b43ec4fb3dc6389377a6',
    'tags': 'yellowstone,landscape',
    'tag_mode': 'all'
})

for photo in r.json()['photos']['photo']:
    file = "data/flickr/" + photo['id'] + '.jpg'
    url = photo_url.substitute(farmid=photo['farm'], serverid=photo['server'], id=photo['id'], secret=photo['secret'])
    if not os.path.isfile(file):
        print "Downloading " + url + " to " + file + "..."
        photo_req = requests.get(url)
        if photo_req.status_code != requests.codes.ok:
            print "Error requesting photo."
            break

        with open(file, 'wb') as fd:
            for chunk in photo_req.iter_content(1024):
                fd.write(chunk)
    else:
        print file + " already exists..."
