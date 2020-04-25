import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to the input image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

(blue, green, red) = image[0][0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(red, green, blue))

image[0,0] = (0,0,255)
(blue, green, red) = image[0,0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(red, green, blue))

corner = image[:100, :100]
cv2.imshow("Corner", corner)
cv2.waitKey(0)

image[:100, :100] = (0,255,0)
cv2.imshow("Updated", image)
cv2.waitKey(0)