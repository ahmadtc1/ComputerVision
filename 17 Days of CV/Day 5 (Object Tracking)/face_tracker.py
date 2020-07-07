import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
import imutils
import time
from collections import deque


#Argument Parsing
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'Deploy' prototxt file")

ap.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")

ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probbability to filter weak detections")

ap.add_argument("-b", "--buffer", type=int, default=64, help="Desired buffer size")

args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

points = deque(maxlen=args["buffer"])

#Start the video stream and let it sleep for 2 seconds to allow it to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:

    #Obtain the frame from the video stream
    frame = vs.read()
    #Resize the frame to be 400 px wide
    frame = imutils.resize(frame, width=400)

    #Extract the height and width of the the frame
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        #Obtain the confidence from the detections
        confidence = detections[0,0,i,2]

        #If the confidence is above our minimum threshold, show the detected face
        if (confidence < args["confidence"]):
            continue

        #Find the x and y bounds for the bounding box representing the found face
        box = detections[0,0,i,3:7] * np.array([width,height,width,height])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)


        center = ((startX + endX) // 2, (startY + endY) // 2)
        points.appendleft(center)

        for i in range(1, len(points)):
            #If either of the tracked points are None, simply continue
            if (points[i - 1] is None or points[i] is None):
                continue

            #Otherwise compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2)
            cv2.line(frame, points[i - 1], points [i], (0, 255, 0), thickness)

        cv2.imshow("Frame", frame)

        #Query a keypress and quit if q is pressed
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("q")):
            break

#Clean up the cv2 windosw and the video stream
cv2.destroyAllWindows()
vs.stop()

