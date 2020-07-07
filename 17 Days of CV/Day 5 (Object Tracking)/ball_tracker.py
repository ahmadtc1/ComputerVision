import cv2
import imutils
from collections import deque
import numpy as np
import argparse
from imutils.video import VideoStream
import time

ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", help="Path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="Maximum buffer size")

args = vars(ap.parse_args())

#define the lower and upper bounds of the color green in the HSV color space
greenLower = (29,86,6)
greenUpper = (64, 255, 255)
points = deque(maxlen=args["buffer"])


#if a video path isn't specified, start a video stream
if not args.get("video", False):
    vs = VideoStream(src=0).start()

else:
    vs = cv2.VideoCapture(args["video"])

#Allow the camera or video time to warm up
time.sleep(2.0)

Smask = None

while True:
    #Grab the current frame
    frame = vs.read()
    #Handle the frame from the VideoStream or VideoCapture
    frame = frame[1] if args.get("Video", False) else frame

    #Scenario where we've reached the end of VideoCapture
    if frame is None:
        break

    #Resize the frame, blur it, and convert color to HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #Construct a mask for the color green
    #Erode and dilate the mask to remove small blobs
    #This mask will capture any objects with the colour green
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #Find the contours in the mask and initialize the center (x,y) of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    if (len(contours) > 0):
        #Find the largest contour if there are multiple contours
        contour = max(contours, key=cv2.contourArea)

        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        center = ( int( M["m10"] / M["m00"] ) , int( M["m01"] / M["m00"] ))

        #only proceed if the radius satisfies a certain size requirement
        if radius > 10:
            #Draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


    points.appendleft(center)

    for i in range(1, len(points)):
        #If either of the tracked points are None, simply continue
        if (points[i - 1] is None or points[i] is None):
            continue

        #Otherwise compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, points[i - 1], points [i], (0, 0, 255), thickness)

    #Show the frame to our screen
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if (key == ord("q")):
        break


#If we're not using a Video File, stop the camera video stream
if not args.get("video", False):
    vs.stop()

#Otherwise release the camera
else:
    vs.release()

#Close all the windows
cv2.destroyAllWindows()



