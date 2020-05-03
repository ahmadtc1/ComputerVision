import cv2
import numpy as np
import imutils
import argparse
import dlib
from imutils import face_utils

#Parse arguments from the command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
args = vars(ap.parse_args())

#Initialize dlib's face detector which is HOG (Histogram of Oriented Gradients) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape-predictor"])

#Load the image and resize it as well as convert to greyscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect the faces in the greyscale image
rects = detector(grey, 1)

for (i, rect) in enumerate(rects):
    
    #Determine facial landmarks for each face
    #Convert the facial landmark coordinates to an np array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    #Convert rectangle to an openCV bounding box (x, y, w, h)
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Face #{}".format(i+1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x,y), 2, (0, 0, 255), -1)

cv2.imshow("Face Landmarked", image)
cv2.waitKey(0)

