import cv2
import argparse
import imutils
from imutils.perspective import four_point_transform
import numpy as np
from imutils import contours

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image to be graded")

args = vars(ap.parse_args())

ANSWER_KEY = {
    0: 1,
    1: 4,
    2: 0,
    3: 3,
    4: 1
}


#load image, convert to grayscale, blur it, then find edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)

cv2.imshow("Edge Detected", edged)
cv2.waitKey(0)

#Find the contours
conts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)

if (len(conts) > 0):
    conts = sorted(conts, key=cv2.contourArea, reverse=True)

    #Loop through the contours to determine the contour of the sheet of paper
    for contour in conts:
        perimeter = cv2.arcLength(contour, True)
        
        approximate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if (len(approximate) == 4):
            documentContour = approximate
            break

#Outline the sheet of paper contour on the image
cv2.drawContours(image, [documentContour], -1, (0,0,255), 3)
cv2.imshow("Contour", image)
cv2.waitKey(0)

#Using the sheet of paper contour, apply a four point transform to get a top-down view
paper = four_point_transform(image, documentContour.reshape(4,2))
warped = four_point_transform(gray, documentContour.reshape(4,2))

cv2.imshow("Image", warped)
cv2.waitKey(0)

#Now thresh this top-down view with Otsu's thresholding method
#This results in a binary image (pixels are either 0x00 or 0xff)
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshed", thresh)
cv2.waitKey(0)

#Once again, find the contours in the now threshed image
conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)
questionContours = []

#Loop through the contours and add them to the questionContours array if they meet our specifications
for contour in conts:
    #Compute the bounding box for the identified contour and compute its aspect ratio
    (x,y,w,h) = cv2.boundingRect(contour)
    aspectRatio = w / float(h)

    

    #To be a selection, the size should be reasonably big and the aspect ratio should be ~ 1 (since it's a circle)
    if (w >= 20 and h >= 20 and aspectRatio >= 0.9 and aspectRatio <= 1.1):
        questionContours.append(contour)

#Show an output image with each of the question contours outlined
output = paper.copy()
for contour in questionContours:
    cv2.drawContours(output, [contour], -1, (0,255,0), 3)

cv2.imshow("Output", output)
cv2.waitKey(0)

questionContours = contours.sort_contours(questionContours, method="top-to-bottom")[0]
correct = 0

#Loop through each of the questionContours with a step of 5 (a new row each time)
for (question, index) in enumerate(np.arange(0, len(questionContours), 5)):
    
    #Sort the contours from left to right for the next 5 contours (each row)
    conts = contours.sort_contours(questionContours[index: index+5])[0]
    bubbled = None

    for (j, c) in enumerate(conts):
        #Construct a mask to reveal only the current "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        #Bitwise and the mask at each contour location and count the nonzero pixels after masking
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        #If we've found a contour with more nonzero pixels (meaning it's bubbled in) store it and its index
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

        color = (0,0,255) #Red in the case that the answer was incorrect
        k = ANSWER_KEY[question]

        #See if the bubbled answer is correct (the index of the found bubbled answer should match the answer key index)
        if (k == bubbled[1]):
            color = (0,255,0)
            correct += 1

        cv2.drawContours(paper, [conts[k]], -1,color, 3)

print("[INFO] Score: {:.2f} %".format((correct / len(ANSWER_KEY) * 100)))
cv2.putText(paper, "{:.2f}%".format((correct / len(ANSWER_KEY) * 100)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

cv2.imshow("Marked", paper)
cv2.waitKey(0)