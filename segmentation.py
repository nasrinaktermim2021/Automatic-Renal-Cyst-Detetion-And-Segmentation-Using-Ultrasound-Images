# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:56:44 2020

@author: nasrin akter
"""

import cv2
#%matplotlib inline

import numpy as np

import os
import matplotlib.pyplot as plt
from glob import glob

import scipy.ndimage
from skimage import morphology
from skimage import measure,color
from skimage.transform import resize
from sklearn.cluster import KMeans
import cv2

 
img= cv2.imread("F:\\thesis\\Automatic renal cysts detection and segementation\\images\\im2.jpg")

width =476 
height = 378
dim = (width, height) 
im2=cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
img1=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
def make_kidneymask(image,img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the kidney
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground 
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the kidney.  
    

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))#sure backgroud

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #   we do another large dilation
    #  in order to fill in and out the kidney mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    print(dilation.dtype)
    print(mask.dtype)
    mm=mask*img
    dilation=np.uint8(dilation)
    ret, markers = cv2.connectedComponents(np.uint8(mm))
    markers = markers+1
   
    markers[dilation==255] = 0
    #print(markers.shape)
    seg=cv2.watershed(image,markers)
    plt.imshow(seg)
    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels ,cmap='gray')
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img
im_mk=make_kidneymask(im2,img1, display=True) 
im_mk=np.uint8(im_mk)
plt.imshow(im_mk, cmap='gray')
#cv2.waitKey(0)
#cv2.destroyAllWindows()