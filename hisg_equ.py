# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:47:45 2020

@author: nasrin akter
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('images/im1.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()

img1 = cv2.imread('denosing image/filter.jpg',0)
hist,bins = np.histogram(img1.flatten(),256,[0,256])
plt.hist(img1.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()
