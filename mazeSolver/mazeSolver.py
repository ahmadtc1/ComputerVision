import cv2
import numpy as np
import argparse
from time import sleep
import sys

# Parse user arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input maze image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
'''
# Handle pngs which have a transparent (black) bg
if (args["image"].rsplit('.')[-1] == 'png'):
    print("{}, {}, {}".format(image.shape[0], image.shape[1], image.shape[2]))
    whiteBg = np.ones(image.shape[:2], dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_and(image, whiteBg)

    cv2.imshow("Image", image)
'''

# Print the initial dimensions of the image for bookkeeping
(height, width) = image.shape[:2]
print("Original image size is: ({}, {})".format(width, height))

# Resize the image so that it has a width of 300 (maintaining aspect ratio)
w = 350.0
ratio = w / image.shape[1]
dimensions = (int(w), int(image.shape[0] * ratio))
image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
print("Resized to : ({}, {})".format(dimensions[0], dimensions[1]))
cv2.imshow("Original", image)

# Convert the image to grayscale and perform a threshold to set all pixels to 0x00 or 0xff
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(T, threshed) = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshed", threshed)

cv2.waitKey(0)

threshed = cv2.dilate(threshed, None, iterations=5)
threshed = cv2.erode(threshed, None, iterations=5)

cv2.imshow("Dilated/Eroded", threshed)


graph = np.ones(threshed.shape[:2], dtype="uint32") * sys.maxsize

