import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

PHOTODIR = "/Users/AhmadTC/Desktop/Pictures:Vids/DSC_5245.JPG"

img_array = cv2.imread(PHOTODIR, cv2.IMREAD_GRAYSCALE)
plt.imshow(img_array, cmap='gray')
plt.show()

print(img_array)
print(img_array.shape)


kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
kernelOne = kernel[0][0]
kernelTwo = kernel[1][0]
kernelThree = kernel[2][0]
kernelFour = kernel[0][1]
kernelFive = kernel[1][1]
kernelSix = kernel[2][1]
kernelSeven = kernel[0][2]
kernelEight = kernel[1][2]
kernelNine = kernel[2][2]

print(kernel.shape)
blurred_img_array = img_array

division_factor = 14
sum = 0

for x in range(2,1598, 3):
    for y in range(2,2398, 5):
        sum += img_array[x-1][y-1] * kernelOne
        sum += img_array[x][y-1] * kernelTwo
        sum += img_array[x+1][y-1] * kernelThree
        sum += img_array[x-1][y] * kernelFour
        sum += img_array[x+1][y] * kernelSix
        sum += img_array[x-1][y+1] * kernelSeven
        sum += img_array[x][y+1] * kernelEight
        sum += img_array[x+1][y+1] * kernelNine

        average = sum / division_factor
        blurred_img_array[x][y] = average

plt.imshow(blurred_img_array)
plt.show()
