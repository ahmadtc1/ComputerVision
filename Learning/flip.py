import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

dim = (image.shape[0], image.shape[1] * 2, 3)
canvas = np.zeros(dim, dtype="uint8")
cv2.imshow("canvas", canvas)


cv2.imshow("Canvas", canvas)

flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)

canvas[ : , : image.shape[1]] = image[:, :]

flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)

flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally and Vertically", flipped)

cv2.waitKey(0)