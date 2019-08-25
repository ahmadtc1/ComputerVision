import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import filters

#Set the directory for the desired photo
FILE_DIRECTORY = "/Users/AhmadTC/Desktop/Pictures:Vids/bridge.jpg"
FILE_DIRECTORY_GENERAL = "/Users/AhmadTC/Desktop/Pictures:Vids"

#Load the image as grayscale into an array
img_array = cv2.imread(FILE_DIRECTORY, cv2.IMREAD_GRAYSCALE)

#Show the image before performing edge detection
plt.imshow(img_array, cmap='gray')
plt.show()

#Apply a gaussian blur to remove high frequency noise
blurred = cv2.GaussianBlur(img_array, (3, 3), 0)

plt.imshow(blurred, cmap='gray')
plt.show()


#Calculating the upper and lower threshold values for Canny edge detection
v = np.median(blurred)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

#perform Canny edge detection using the calculated threshold values
edges = cv2.Canny(blurred, lower, upper)

#Display the edge detected image
plt.imshow(edges, cmap='gray')
plt.show()