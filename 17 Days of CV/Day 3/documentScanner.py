import cv2
import imutils
import numpy as np
import argparse
from skimage.filters import threshold_local
from pyimagesearch.transform import four_point_transform

#Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the desired input image")

args = vars(ap.parse_args())

#Read in the image and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
original = image.copy()
image = imutils.resize(image, height=500)

#Convert image to greyscale. Blur the image. Perform edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

cv2.imshow("Image", image)
cv2.waitKey(0)


print("STEP 1: Edge Detection")
cv2.imshow("Edged", edged)
cv2.waitKey(0)

cv2.destroyAllWindows()

#Find the contours in the edged image and only track the largest one
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]

#Loop over the contours
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approximate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if (len(approximate) == 4):
        screenContour = approximate
        break

#Show the contour for the piece of paper
print("STEP 2: Find contours of the paper")
cv2.drawContours(image, [screenContour], -1, (0,255,0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Apply 4 point transform for top-down view of image
warped = four_point_transform(original, screenContour.reshape(4,2) * ratio)

#Convert warped image to greyscale, then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(original, height=650))
cv2.waitKey(0)

cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)

cv2.destroyAllWindows()