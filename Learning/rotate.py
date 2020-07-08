from argparse import ArgumentDefaultsHelpFormatter
import cv2
#import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to desired input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(height, width) = image.shape[:2]
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Rotated by 45 degrees", rotated)

M = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated = cv2.warpAffine(image, M, (width, height))
cv2.imshow("Rotated by -90 degrees", rotated)

cv2.waitKey(0)

def rotate(image, angle, center=None, scale=1.0):
    (height, width) = image.shape[:2]
    
    if center is None:
        center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (width, height))

    return rotated

img = cv2.imread(args["image"])
rotated = rotate(image, -27)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
