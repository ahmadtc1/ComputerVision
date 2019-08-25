import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import filters

#Set the directory for the desired photo
FILE_DIRECTORY = "/Users/AhmadTC/Desktop/Pictures:Vids/cube-01-01-01.jpg"
FILE_DIRECTORY_GENERAL = "/Users/AhmadTC/Desktop/Pictures:Vids"

#Load the image as a grayscale image into an array
img_array = cv2.imread(FILE_DIRECTORY, cv2.IMREAD_GRAYSCALE)

#Show the image before performing edge detection
plt.imshow(img_array, cmap='gray')
plt.show()

#Obtain the otsu threshold to be used in canny edge detection
otsu = filters.threshold_otsu(img_array)


#perform Canny edge detection using the otsu threshold
edges = cv2.Canny(img_array, otsu*0.5, otsu)

#Display the edge detected image
plt.imshow(edges, cmap='gray')
plt.show()