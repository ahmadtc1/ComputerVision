import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

#Set the path to the desired photo
PHOTODIR = "/Users/AhmadTC/Desktop/Pictures:Vids/DSC_5245.JPG"

#Load the image as a grayscale image array
img_array = cv2.imread(PHOTODIR, cv2.IMREAD_GRAYSCALE)

#Show the image before the blur
plt.imshow(img_array, cmap='gray')
plt.show()

#Set the gaussian blur kernel and assign each value of the kernel to a variable for later convenience
kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
kernelOne = kernel[0][0]
kernelTwo = kernel[1][0]
kernelThree = kernel[2][0]
kernelFour = kernel[0][1]
kernelFive = kernel[1][1]
kernelSix = kernel[2][1]
kernelSeven = kernel[0][2]
kernelEight = kernel[1][2]
kernelNine = kernel[2][2]

#Copy the image array into a new array because we don't want to alter the original image
blurred_img_array = img_array


division_factor = 14 #Sum of the values in the kernel
sum = 0

for x in range(1, img_array.shape[0] - 1):
    for y in range(1, img_array.shape[1] - 1):
        #Sum up the values of each surrounding pixel multiplied by their respective kernel value
        sum = 0
        sum += img_array[x-1][y-1] * kernelOne
        sum += img_array[x][y-1] * kernelTwo
        sum += img_array[x+1][y-1] * kernelThree
        sum += img_array[x-1][y] * kernelFour
        sum += img_array[x][y] * kernelFive
        sum += img_array[x+1][y] * kernelSix
        sum += img_array[x-1][y+1] * kernelSeven
        sum += img_array[x][y+1] * kernelEight
        sum += img_array[x+1][y+1] * kernelNine

        #Calculate the average value by dividing the sum by the division factor
        average = sum / division_factor
        blurred_img_array[x][y] = average

#Show the blurred image
plt.imshow(blurred_img_array, cmap='gray')
plt.show()