#!/usr/bin/env python
import urllib2

nps_site = "http://www.nps.gov/features/yell/slidefile/"
imageDirectory = "data/nps/"

# Get the index page of a URL
def getPage(url):
    return url.rfind('/') + 1

# Return list of URLs 
def readPageForURLs(site, html):
    urls = []
    start = 0
    urlPos = html.find("<a ")
    while urlPos >= 0:
        startIndex = html.find('"', urlPos)
        endIndex = html.find('"', startIndex + 1)

        # Only include relative URLs. Also, exluce images
        if html[startIndex + 1:startIndex + 5] != 'http' and \
            html[startIndex + 1:startIndex + 7] != 'Images':
            urls.append(site + html[startIndex + 1:endIndex])
        urlPos = html.find("<a ", urlPos + 1)
    return urls

# Return list of image links
def readPageForIMGs(site, html):
    imgs = []
    start = 0
    imgPos = html.find("Images/")
    while imgPos >= 0:
        endIndex = html.find('"', imgPos)

        imgs.append(site + html[imgPos:endIndex])
        imgPos = html.find("Images/", imgPos + 1)
    return imgs

# Recursively search the given site for images.
def retrieveImages(site, page, foundSites, images):
    if site + page in foundSites:
        return
    foundSites.add(site + page)

    try:
        response = urllib2.urlopen(site + page)
        html = response.read()

        # Read the urls and the images of the page
        newURLs = readPageForURLs(site, html)
        newIMGs = readPageForIMGs(site, html)
        images.extend(newIMGs)

        # Recursively search the new URLs
        for newURL in newURLs:
            pageIndex = getPage(newURL)
            newSite = newURL[:pageIndex]
            newPage = newURL[pageIndex:]
            retrieveImages(newSite, newPage, foundSites, images)
    except:
        print 'Could not read:', site + page

# Downloads the image with this URL.
def downloadImage(image):
    try:
        response = urllib2.urlopen(image)
        binary = response.read()
        imageIndex = getPage(image)
        outputFile = image[imageIndex:]
        f = open(imageDirectory + outputFile, 'w')
        f.write(binary)
        f.close()
    except:
        print 'Could not read image:', image

images = []
retrieveImages(nps_site, "", set(), images)
print "Found", len(images), "images"
for image in images:
    downloadImage(image)
