import cv2
import numpy as np
import imutils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


img = cv2.imread(args["image"])
cv2.imshow("Original", img)

#translation matrix, must be a floating point matrix
M = np.float32([ [1, 0, 25], [0, 1, 50] ])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Shifted down and right", shifted)

M = np.float32([ [1, 0, -50], [0, 1, -90] ])
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Shifted up and left", shifted)

cv2.waitKey(0)