import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
cv2.imshow("Original", image)

#Compute the ratio if converting to 150
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)

def resize(image, width = None, height = None, inter=cv2.INTER_AREA):
    
    if (width == None and height == None):
        return image

    elif (width == None and height is not None):
        r = float(height) / image.shape[0] 
        dim = (int(image.shape[1] * r), height)
        return cv2.resize(image, dim, interpolation=inter)

    elif (height == None and width is not None):
        r =  float(width) / image.shape[1]
        dim = (width, int(image.shape[1] * r))
        return cv2.resize(image, dim, interpolation=inter)

    elif (height is not None and width is not None):
        return cv2.resize(image, (width, height), interpolation=inter)


resized = resize(image, 500, 2000)
cv2.imshow("resized", resized)
cv2.waitKey(0)