# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:55:25 2020

@author: nasrin akter
"""

import cv2 as cv
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load original image
originalImage = cv.imread("images/im2.jpg")
originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2RGB)
reshapedImage = np.float32(originalImage.reshape(-1, 3))
numberOfClusters = 2

stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)

ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)

clusters = np.uint8(clusters)

intermediateImage = clusters[labels.flatten()]
clusteredImage = intermediateImage.reshape((originalImage.shape))

cv.imwrite("clusteredImage.jpg", clusteredImage)

# Remove 1 cluster from image and apply canny edge detection
removedCluster = 1



params = cv2.SimpleBlobDetector_Params()

# Define thresholds
#Can define thresholdStep. See documentation. 
params.minThreshold = 10
params.maxThreshold = 87

# Filter by Area.
params.filterByArea = True
params.minArea = 150
params.maxArea = 50000

# Filter by Color (black=0)
params.filterByColor = True  #Set true for cast_iron as we'll be detecting black regions
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2
params.maxCircularity = 3

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01
params.maxConvexity = 4

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 4

# Distance Between Blobs
params.minDistBetweenBlobs = .1

# Setup the detector with parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(originalImage)

print("Number of cyst detected are : ", len(keypoints))

# Draw blobs
img_with_blobs = cv2.drawKeypoints(originalImage, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_blobs)


img_with_blobs = np.copy(originalImage).reshape((-1, 3))
img_with_blobs[labels.flatten() == removedCluster] = [0, 0, 0]

img_with_blobs = cv.Canny(img_with_blobs,100,200).reshape(originalImage.shape)
cv.imshow("cannyImage.jpg", img_with_blobs)
cv.waitKey()
cv.destroyAllWindows()

initialContoursImage = np.copy(img_with_blobs)
imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(imgray, 50, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(initialContoursImage, contours, -1, (0,0,255), cv.CHAIN_APPROX_SIMPLE)
cv.imshow("initialContoursImage.jpg", initialContoursImage)
cv.waitKey()
cv.destroyAllWindows()


cnt = contours[0]
largest_area=0
index = 0
for contour in contours:
    if index > 0:
        area = cv.contourArea(contour)
        if (area>largest_area):
            largest_area=area
            cnt = contours[index]
    index = index + 1

biggestContourImage = np.copy(originalImage)
cv.drawContours(biggestContourImage, [cnt], -1, (0,0,255), 3)
cv.imshow("biggestContourImage.jpg", biggestContourImage)
cv.waitKey()
cv.destroyAllWindows()