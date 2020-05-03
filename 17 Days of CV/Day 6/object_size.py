import cv2
import imutils
import argparse
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np

#Define a function to compute the (x,y) midpoint given two points A and B
def midPoint(pointA, pointB):
    return ( ((pointA[0] + pointB[0]) // 2) , ((pointA[1] + pointB[1]) // 2) )

#Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="The width of the leftmost object in the image (in inches)")
args = vars(ap.parse_args())

#load the image, convert it to grayscale, and perform a slight gaussian blur
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#Edge detect and then dilate and erode to close gaps in object edges
edged = cv2.Canny(blurred, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#Sort the contours from left to right
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

for c in cnts:
    if cv2.contourArea(c)  < 500:
        continue

    #Compute the contour's rotated bounding box
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    #Order the points topleft, topright, bottomright, bottomleft
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0,255,0), 2)

    #Loop over and draw the original points
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0,0,255), -1)

    #Unpack the ordered bounding box and compute the midpoints for the top and bottom edges
    (topleft, topright, bottomright, bottomleft) = box
    (tltrX, tltrY) = midPoint(topleft, topright)
    (blbrX, blbrY) = midPoint(bottomleft, bottomright)

    #Compute the midpoints between the left and right edges
    (tlblX, tlblY) = midPoint(topleft, bottomleft)
    (trbrX, trbrY) = midPoint(topright, bottomright)

    #Draw the midpoints on the images
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255,0,0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255,0,0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255,0,0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255,0,0), -1)

    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)

    height = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    #Compute the ratio of pixels to the supplied metric
    if pixelsPerMetric is None:
        pixelsPerMetric = width / args["width"]

    #Compute the dimensions now with the pixelsPerMetric ratio
    dimensionsA = height / pixelsPerMetric
    dimensionsB = width / pixelsPerMetric

    #Draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimensionsA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimensionsB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    cv2.imshow("Midpointed", orig)
    cv2.waitKey(0)
