import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to input image file")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

print("height: {} pixels".format(image.shape[0]))

print("width: {} pixels".format(image.shape[1]))

print("channels: {}".format(image.shape[0]))

cv2.imshow("Image", image)

cv2.waitKey(0)


cv2.imwrite("newImage.jpg", image)