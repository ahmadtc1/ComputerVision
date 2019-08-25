import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#Set the directory for the desired photo
FILE_DIRECTORY = "/Users/AhmadTC/Desktop/Pictures:Vids/cube-01-01-01.jpg"
FILE_DIRECTORY_GENERAL = "/Users/AhmadTC/Desktop/Pictures:Vids"

#Load the image as a grayscale image into an array
img_array = cv2.imread(FILE_DIRECTORY, cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array, cmap='gray')
plt.show()
sum = 0
total = img_array.shape[0] * img_array.shape[1]

for x in range(img_array.shape[0]):
    for y in range(img_array.shape[1]):
        sum += img_array[x][y]


mean = sum / total

#perform Canny edge detection using openCV
edges = cv2.Canny(img_array, mean*0.75, mean*1.25)

#Display the edge detected image
plt.imshow(edges, cmap='gray')
plt.show()